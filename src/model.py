import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import GRIDS     # >>> [YOLOv3-NEW]
# >>> [YOLOv10-CHANGED] ANCHORS / anchor_tensor / split_anchors_by_scale are no
# longer imported: YOLOv10 is ANCHOR-FREE (inherited from YOLOv8). Every grid
# cell centre is a point that predicts its distances to the four box sides.

# >>> [YOLOv3-NEW] written into every checkpoint, so a checkpoint from another
# branch fails with a sentence instead of 40 lines of state_dict key errors.
ARCH = "yolov10"     # >>> [YOLOv10-CHANGED]

# >>> [YOLOv10-NEW] Distribution Focal Loss bins (YOLOv8 / YOLOv10 default).
# Each box side is predicted as a probability distribution over 0..15 grid
# cells instead of one number; the side length is that distribution's mean.
REG_MAX = 16


# YOLOv4 put Mish in the backbone, YOLOv5 moved everything to SiLU; YOLOv8 /
# YOLOv10 keep SiLU. The segmentation head still uses the "leaky" default.
def activation(name="leaky"):
    if name == "mish":
        return nn.Mish(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "none":     # >>> [YOLOv10-NEW] SCDown / attention use linear convs
        return nn.Identity()
    return nn.LeakyReLU(0.1, inplace=True)


def conv_block(channel_in, channel_out, k=3, s=1, act="leaky", p=None, g=1):     # >>> [YOLOv10-CHANGED] added `g` (groups)
    # s=1 only thickening, s=2 shrinking and thickening
    return nn.Sequential(
        # Converting the picture into thinkened and shrinked picture
        nn.Conv2d(channel_in, channel_out, k, stride=s, padding=(k // 2) if p is None else p,
                  groups=g, bias=False),
        nn.BatchNorm2d(channel_out),
        activation(act)
    )


# ---------------------------------------------------------------------------
# >>> [YOLOv10-NEW] building blocks
# ---------------------------------------------------------------------------

class Bottleneck(nn.Module):
    # YOLOv8's bottleneck: 3x3 -> 3x3, plus the input when shortcut=True.
    # (YOLOv5's was 1x1 -> 3x3.)
    def __init__(self, channels, shortcut=True, act="silu"):
        super().__init__()
        self.net = nn.Sequential(conv_block(channels, channels, k=3, act=act),
                                 conv_block(channels, channels, k=3, act=act))
        self.shortcut = shortcut

    def forward(self, x):
        y = self.net(x)
        return x + y if self.shortcut else y


class C2f(nn.Module):
    # YOLOv8's replacement for YOLOv5's C3 (kept by YOLOv10).
    #
    #   x --1x1--> split --+----------------------------------------+
    #                      +--> block1 --+--> block2 --+--> ...     |
    #                                    |             |            |
    #                      concat(every intermediate output) --1x1--> out
    #
    # C3 only kept the LAST block's output; C2f concatenates every block's
    # output, so the final 1x1 sees features at every depth and the gradient
    # has a short path to each block (the "f" is for the ELAN-style fan-out).
    def __init__(self, channel_in, channel_out, n=1, shortcut=False, act="silu", block=None):
        super().__init__()
        self.c = channel_out // 2
        self.cv1 = conv_block(channel_in, 2 * self.c, k=1, act=act)
        self.cv2 = conv_block((2 + n) * self.c, channel_out, k=1, act=act)
        make = block or (lambda c: Bottleneck(c, shortcut, act))
        self.m = nn.ModuleList(make(self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(th.cat(y, dim=1))


class SCDown(nn.Module):
    # YOLOv10's "spatial-channel decoupled downsampling".
    # A normal stride-2 3x3 conv changes the resolution AND the channel count
    # in one dense op: k*k*C_in*C_out weights. SCDown does the two jobs
    # separately -- a 1x1 for channels, then a DEPTHWISE stride-2 3x3 for
    # resolution -- which costs C_in*C_out + k*k*C_out, and keeps more
    # information through the downsample (the paper's ablation).
    def __init__(self, channel_in, channel_out, k=3, s=2, act="silu"):
        super().__init__()
        self.cv1 = conv_block(channel_in, channel_out, k=1, act=act)
        self.cv2 = conv_block(channel_out, channel_out, k=k, s=s, g=channel_out, act="none")

    def forward(self, x):
        return self.cv2(self.cv1(x))


class RepVGGDW(nn.Module):
    # YOLOv10's large-kernel depthwise conv: a 7x7 and a 3x3 depthwise branch
    # added together. The 3x3 branch helps optimisation early on; the two can
    # be folded into one 7x7 for deployment (not done here, it is only a speed
    # optimisation and the output is identical).
    def __init__(self, channels):
        super().__init__()
        self.conv7 = conv_block(channels, channels, k=7, g=channels, act="none")
        self.conv3 = conv_block(channels, channels, k=3, g=channels, act="none")
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv7(x) + self.conv3(x))


class CIB(nn.Module):
    # YOLOv10's "compact inverted block": depthwise 3x3 for space, pointwise
    # 1x1 for channels, alternated. YOLOv10's rank analysis found that the
    # deepest stages of YOLOv8 are redundant (low-rank weights), so it swaps
    # their dense 3x3 bottlenecks for this much cheaper block.
    # lk=True uses the 7x7 RepVGGDW in the middle -- a bigger receptive field
    # for almost no cost, which the paper uses in the deep stage of small models.
    def __init__(self, channel_in, channel_out, shortcut=True, e=0.5, lk=False, act="silu"):
        super().__init__()
        hidden = int(channel_out * e)
        self.net = nn.Sequential(
            conv_block(channel_in, channel_in, k=3, g=channel_in, act=act),
            conv_block(channel_in, 2 * hidden, k=1, act=act),
            RepVGGDW(2 * hidden) if lk else conv_block(2 * hidden, 2 * hidden, k=3, g=2 * hidden, act=act),
            conv_block(2 * hidden, channel_out, k=1, act=act),
            conv_block(channel_out, channel_out, k=3, g=channel_out, act=act),
        )
        self.add = shortcut and channel_in == channel_out

    def forward(self, x):
        y = self.net(x)
        return x + y if self.add else y


def C2fCIB(channel_in, channel_out, n=1, shortcut=True, lk=False):
    # YOLOv10: C2f with CIB blocks instead of bottlenecks (deep stages only)
    return C2f(channel_in, channel_out, n, block=lambda c: CIB(c, c, shortcut, e=1.0, lk=lk))


class SPPF(nn.Module):
    # From YOLOv5: three 5x5 max-pools in series == SPP(5, 9, 13), ~2x faster.
    def __init__(self, channel_in, channel_out, k=5, act="silu"):
        super().__init__()
        hidden = channel_in // 2
        self.cv1 = conv_block(channel_in, hidden, k=1, act=act)
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)
        self.cv2 = conv_block(hidden * 4, channel_out, k=1, act=act)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        return self.cv2(th.cat([x, y1, y2, self.pool(y2)], dim=1))


class Attention(nn.Module):
    # Multi-head self-attention over the cells of a feature map, as in
    # YOLOv10's PSA. A depthwise 3x3 on the values acts as the positional
    # encoding, so no learned position table is needed and any map size works.
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        self.qkv = conv_block(dim, dim + 2 * self.key_dim * num_heads, k=1, act="none")
        self.proj = conv_block(dim, dim, k=1, act="none")
        self.pe = conv_block(dim, dim, k=3, g=dim, act="none")

    def forward(self, x):
        N, C, H, W = x.shape
        q, k, v = self.qkv(x).view(N, self.num_heads, 2 * self.key_dim + self.head_dim, H * W).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn = ((q.transpose(-2, -1) @ k) * self.scale).softmax(dim=-1)        # [N, heads, HW, HW]
        out = (v @ attn.transpose(-2, -1)).view(N, C, H, W) + self.pe(v.reshape(N, C, H, W))
        return self.proj(out)


class PSA(nn.Module):
    # YOLOv10's "partial self-attention". Self-attention is the best tool for
    # GLOBAL context (where is the hand relative to the whole frame) but it
    # is quadratic in the number of cells, so YOLOv10 uses it once, on the
    # smallest map, and only on HALF the channels: the other half passes
    # through untouched and is concatenated back. On our 7x7 map that is 49
    # tokens -- attention there is nearly free.
    def __init__(self, channels, e=0.5):
        super().__init__()
        self.c = int(channels * e)
        self.cv1 = conv_block(channels, 2 * self.c, k=1, act="silu")
        self.cv2 = conv_block(2 * self.c, channels, k=1, act="silu")
        self.attn = Attention(self.c, num_heads=max(1, self.c // 64), attn_ratio=0.5)
        self.ffn = nn.Sequential(conv_block(self.c, 2 * self.c, k=1, act="silu"),
                                 conv_block(2 * self.c, self.c, k=1, act="none"))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(th.cat([a, b], dim=1))


class Backbone(nn.Module):
    # >>> [YOLOv10-CHANGED] YOLOv10 backbone (YOLOv8 layout + YOLOv10 blocks):
    #   s1        3x3 stride-2 conv stem (YOLOv8 went back from 6x6 to 3x3)
    #   s2 .. s4  stride-2 conv + C2f
    #   s5, s6    SCDown + C2f / C2fCIB(lk=True) in the deepest stage
    # SCDown and CIB only in the deep stages is the paper's rank-guided design:
    # that is where the dense blocks were found to be redundant.
    # Resolutions / channels / outputs are unchanged (f56, f28, f14, f7), so
    # the segmentation and classification heads see the same tensors.
    # depths (1, 1, 2, 2, 1): YOLOv8/10's 3 : 6 : 6 : 3 pattern at small-model depth.
    def __init__(self, in_channels=4, depths=(1, 1, 2, 2, 1)):
        super().__init__()
        # Reducing the pixels into 7x7 (448 / 2^6)
        # 7x7 being the box of determining the size of the grid cell
        self.s1 = conv_block(in_channels, 32, k=3, s=2, act="silu")
        self.s2 = nn.Sequential(conv_block(32, 64, s=2, act="silu"), C2f(64, 64, depths[0], shortcut=True))
        self.s3 = nn.Sequential(conv_block(64, 128, s=2, act="silu"), C2f(128, 128, depths[1], shortcut=True))
        self.s4 = nn.Sequential(conv_block(128, 256, s=2, act="silu"), C2f(256, 256, depths[2], shortcut=True))
        self.s5 = nn.Sequential(SCDown(256, 512), C2f(512, 512, depths[3], shortcut=True))
        self.s6 = nn.Sequential(SCDown(512, 1024), C2fCIB(1024, 1024, depths[4], shortcut=True, lk=True))


    def forward(self, x):
        # Forward pass to get data from size 7 to 56 and later hand to SegmentationHead to help to classify the border
        # Mainly using s3 to s5 to calibrate the shape, therefore keeping the record all these data and ready to share backbone
        f56 = self.s3(self.s2(self.s1(x)))
        f28 = self.s4(f56)
        f14 = self.s5(f28)
        f7 = self.s6(f14)
        return f56, f28, f14, f7


class PANNeck(nn.Module):
    # >>> [YOLOv10-CHANGED] YOLOv10's neck: SPPF -> PSA on the 7x7 map, then
    # the YOLOv8 PAN (top-down, then bottom-up, concat merges) built from C2f,
    # with SCDown + C2fCIB on the last, deepest bottom-up step.
    # (In YOLOv10 SPPF + PSA sit at the end of the backbone. They are kept in
    # the neck here, as on the earlier branches, so the classification and
    # segmentation heads keep reading the plain backbone f7.)
    #
    #   f7 -> SPPF -> PSA (x7) -----------------------------------> concat -> C2fCIB -> n7  (7x7)
    #                  |                                               ^
    #                  up x2                                           SCDown
    #                  v                                               |
    #   f14 ------> concat -> C2f -> t14 ------------------------> concat -> C2f -> n14   (14x14)
    #                                 |                    ^
    #                                 up x2                3x3 s2
    #                                 v                    |
    #   f28 -----------------------> concat -> C2f -> n28 -+                           (28x28)
    def __init__(self, n=1):
        super().__init__()
        self.sppf = SPPF(1024, 1024)
        self.psa = PSA(1024)
        self.td14 = C2f(1024 + 512, 512, n)
        self.td28 = C2f(512 + 256, 256, n)
        self.down28 = conv_block(256, 256, s=2, act="silu")
        self.bu14 = C2f(256 + 512, 512, n)
        self.down14 = SCDown(512, 512)
        self.bu7 = C2fCIB(512 + 1024, 1024, n, shortcut=True, lk=True)
        self.out_channels = (1024, 512, 256)    # n7, n14, n28

    def forward(self, f28, f14, f7):
        x7 = self.psa(self.sppf(f7))
        t14 = self.td14(th.cat([F.interpolate(x7, scale_factor=2), f14], dim=1))
        n28 = self.td28(th.cat([F.interpolate(t14, scale_factor=2), f28], dim=1))
        n14 = self.bu14(th.cat([self.down28(n28), t14], dim=1))
        n7 = self.bu7(th.cat([self.down14(n14), x7], dim=1))
        return n7, n14, n28


class DetectionHead(nn.Module):
    # >>> [YOLOv10-CHANGED] rewritten. History of this class:
    #   YOLOv1  B free boxes per 7x7 cell
    #   YOLOv2  one box per ANCHOR, decoded as a correction to the anchor
    #   YOLOv3  three grids, anchors split over them
    #   YOLOv5  bounded decode, objectness trained towards IoU
    #   YOLOv10 (this) -- ANCHOR-FREE, DECOUPLED, DUAL-HEAD, NMS-FREE:
    #
    # * Anchor-free (from YOLOv8). Every cell centre (col + 0.5, row + 0.5) is
    #   a point; it predicts the distances (l, t, r, b) to the four box sides.
    #   No priors to fit, so no k-means and no autoanchor any more.
    # * DFL box regression (from YOLOv8). Each distance is a softmax over
    #   REG_MAX bins (0..15 cells); the distance is its expected value. A
    #   distribution can say "the edge is between 2 and 3 cells away", which
    #   suits blurry hand edges better than one regressed number.
    # * Decoupled head (from YOLOv8). Separate conv branches for the box and
    #   for the score. YOLOv10 makes the score branch LIGHTWEIGHT (depthwise
    #   3x3 + pointwise 1x1, twice): its analysis found the classification
    #   branch needs far less capacity than the regression one.
    # * No separate objectness. The score is the "hand" class probability,
    #   trained towards an IoU-aware target (task alignment, see train.py).
    # * Dual heads ("consistent dual assignments"):
    #     one-to-many (o2m)  trained with the top-10 points per hand -> rich
    #                        supervision for the backbone
    #     one-to-one  (o2o)  an identical copy trained with the top-1 point per
    #                        hand -> exactly one confident box per hand, so no
    #                        NMS is needed at inference
    #   Both use the SAME matching metric, so the o2o head's best point is also
    #   the o2m head's best point. The o2o head reads DETACHED features: only
    #   the o2m head trains the backbone, as in YOLOv10.
    #
    # Output, one tensor [N, M, 5 + 2 * (4 * REG_MAX + 1)], M = 28^2 + 14^2 + 7^2:
    #   [..., 0:5]  o2o boxes, decoded: (cx, cy, w, h, score) in [0, 1]
    #               -> what evaluate.py / visualize.py / the demo use
    #   [..., 5:5+K]  o2m raw: 4*REG_MAX side logits + 1 score logit   (loss only)
    #   [..., 5+K:]   o2o raw, same layout                             (loss only)
    # A single tensor (not a dict) so _forward_fp32 and the DataLoader-style
    # tuple handling in train.py keep working unchanged.
    def __init__(self, in_channels=(1024, 512, 256), grids=GRIDS, reg_max=REG_MAX):
        super().__init__()
        self.grids = tuple(grids)
        self.reg_max = reg_max
        self.n_raw = 4 * reg_max + 1
        c_box = max(16, in_channels[-1] // 4, reg_max * 4)
        c_cls = max(in_channels[-1], 1)
        self.o2m_box = nn.ModuleList(self._box_branch(c, c_box) for c in in_channels)
        self.o2m_cls = nn.ModuleList(self._cls_branch(c, c_cls) for c in in_channels)
        self.o2o_box = nn.ModuleList(self._box_branch(c, c_box) for c in in_channels)
        self.o2o_cls = nn.ModuleList(self._cls_branch(c, c_cls) for c in in_channels)
        self._init_bias()

        # Anchor POINTS: centre of every cell of every grid, coarse grid first,
        # normalised to [0, 1], plus the grid size each point belongs to.
        points, grid_of = [], []
        for S in self.grids:
            r, c = th.meshgrid(th.arange(S), th.arange(S), indexing="ij")
            points.append(th.stack([(c.flatten() + 0.5) / S, (r.flatten() + 0.5) / S], dim=1))
            grid_of.append(th.full((S * S,), S))
        self.register_buffer("points", th.cat(points).float(), persistent=False)
        self.register_buffer("point_grid", th.cat(grid_of), persistent=False)
        self.register_buffer("proj", th.arange(reg_max).float(), persistent=False)
        self.n_boxes = self.points.size(0)

    def _box_branch(self, c, c_box):
        return nn.Sequential(conv_block(c, c_box, k=3, act="silu"),
                             conv_block(c_box, c_box, k=3, act="silu"),
                             nn.Conv2d(c_box, 4 * self.reg_max, kernel_size=1))

    @staticmethod
    def _cls_branch(c, c_cls):
        # YOLOv10's lightweight classification branch
        return nn.Sequential(conv_block(c, c, k=3, g=c, act="silu"), conv_block(c, c_cls, k=1, act="silu"),
                             conv_block(c_cls, c_cls, k=3, g=c_cls, act="silu"), conv_block(c_cls, c_cls, k=1, act="silu"),
                             nn.Conv2d(c_cls, 1, kernel_size=1))

    def _init_bias(self):
        # YOLOv8's head init: box logits start flat (bias 1); the score starts
        # at the prior rate log(5 / cells) -- one hand per image, ~5 of the
        # cells around it are positives early in training.
        for boxes, scores in ((self.o2m_box, self.o2m_cls), (self.o2o_box, self.o2o_cls)):
            for b, c, S in zip(boxes, scores, self.grids):
                with th.no_grad():
                    b[-1].bias.fill_(1.0)
                    c[-1].bias.fill_(math.log(5.0 / S ** 2))

    def layout(self):
        """What the loss needs: point centres, which grid each point is on."""
        return {"points": self.points, "grid": self.point_grid, "reg_max": self.reg_max}

    def decode(self, raw):
        """raw [N, M, 4*REG_MAX + 1] -> (cx, cy, w, h, score), all normalised."""
        raw = raw.float()
        N, M, _ = raw.shape
        dist = raw[..., :4 * self.reg_max].view(N, M, 4, self.reg_max).softmax(dim=-1) @ self.proj
        dist = dist / self.point_grid.view(1, M, 1).to(dist.dtype)              # cells -> image units
        px, py = self.points[:, 0], self.points[:, 1]
        x1, y1 = px - dist[..., 0], py - dist[..., 1]
        x2, y2 = px + dist[..., 2], py + dist[..., 3]
        score = th.sigmoid(raw[..., -1])
        return th.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, score], dim=-1)

    @staticmethod
    def _run(features, boxes, scores):
        out = []
        for x, b, c in zip(features, boxes, scores):
            y = th.cat([b(x), c(x)], dim=1)                                   # [N, 4R+1, S, S]
            out.append(y.flatten(2).transpose(1, 2))                          # [N, S*S, 4R+1]
        return th.cat(out, dim=1)

    def forward(self, features):
        # features: (n7, n14, n28) from the neck, coarse first like GRIDS
        for x, S in zip(features, self.grids):
            assert x.shape[-1] == S, f"expected a {S}x{S} map, got {tuple(x.shape)}"
        o2m = self._run(features, self.o2m_box, self.o2m_cls)
        o2o = self._run([x.detach() for x in features], self.o2o_box, self.o2o_cls)
        return th.cat([self.decode(o2o), o2m.float(), o2o.float()], dim=-1)


class ClassificationHead(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        # To classify which sign is the possible class we want (returns logit)
        self.net = nn.Sequential(
            nn.Linear(2048, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )


    def forward(self, f7):
        # Extracting from each block whether is contains the object
        avg = F.adaptive_avg_pool2d(f7, 1).flatten(1)
        mx = F.adaptive_max_pool2d(f7, 1).flatten(1)
        return self.net(th.cat([avg, mx], dim=1))
    
class SegmentationHead(nn.Module):
    # Resizing back to input image size with suitable correction and modification
    def __init__(self):
        super().__init__()
        # Sticking back the channels
        self.up1 = conv_block(1024 + 512, 256)
        self.up2 = conv_block(256 + 256, 128)
        self.up3 = conv_block(128 + 128, 64)
        # Converting the all 64 weight per pixel into one number
        self.out = nn.Conv2d(64, 1, kernel_size=1)


    def forward(self, f56, f28, f14, f7):
        # Enlarging by scale factor 2
        x = F.interpolate(f7, scale_factor=2)
        # Following the channel(dimension 1) extend on it e.g. 1024 + 512 = 1536
        x = self.up1(th.cat([x, f14], dim=1))
        x = F.interpolate(x, scale_factor=2)
        x = self.up2(th.cat([x, f28], dim=1))
        x = F.interpolate(x, scale_factor=2)
        x = self.up3(th.cat([x, f56], dim=1))
        # Collapse 64 channels -> 1 BEFORE the upsample, not after. A 1x1 conv is a
        # per-pixel linear map and bilinear interpolation is a weighted average whose
        # weights sum to 1, so the two commute exactly -- same numbers out. Doing it
        # in this order means we never materialise the [N, 64, 448, 448] tensor
        # (822 MB at batch 16, plus the same again for its gradient); the upsample
        # now runs on a single channel instead.
        x = self.out(x)
        return F.interpolate(x, size=(448, 448), mode="bilinear", align_corners=False)
    

class HandGestureNet(nn.Module):
    def __init__(self, in_channels=5, n_classes=10, B=2, anchors=None):
        # >>> [YOLOv10-CHANGED] `B` and `anchors` are accepted and ignored so old
        # call sites keep working; the detector is anchor-free now.
        super().__init__()
        self.backbone = Backbone(in_channels)
        # The neck only feeds detection. Segmentation keeps its own U-Net
        # decoder and classification keeps reading f7.
        self.neck = PANNeck()
        self.detecting_head = DetectionHead(in_channels=self.neck.out_channels)
        self.classification_head = ClassificationHead(n_classes)
        self.segmentation_head = SegmentationHead()

    
    def forward(self, x):
        f56, f28, f14, f7 = self.backbone(x)
        return (self.detecting_head(self.neck(f28, f14, f7)),
                self.segmentation_head(f56, f28, f14, f7),
                self.classification_head(f7)
                )


def load_handgesture_net(checkpoint, in_channels=4, n_classes=10, device=None):
    """
    Rebuild the network and pour the trained weights in.

    >>> [YOLOv10-CHANGED] no anchors to read back any more (anchor-free), but
    the architecture check stays: a checkpoint from yolov2..yolov5 has a
    completely different detection head.
    """
    arch = checkpoint.get("arch")
    if arch != ARCH:
        raise RuntimeError(
            f"This checkpoint was trained with arch={arch!r} (None = YOLOv2 or "
            f"older) but this branch builds {ARCH!r}. The detection heads are "
            f"different networks; evaluate it on the branch it was trained on, "
            f"or retrain here.")
    model = HandGestureNet(in_channels=in_channels, n_classes=n_classes)
    model.load_state_dict(checkpoint["model_state"])
    if device is not None:
        model = model.to(device)
    return model
