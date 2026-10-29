import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math

# >>> [YOLOv2-NEW] anchor priors live in utils.py so the loss, the head and
# kmeans_anchors.py all read the same numbers.
from utils import ANCHORS, anchor_tensor
from utils import GRIDS, split_anchors_by_scale     # >>> [YOLOv3-NEW]

# >>> [YOLOv3-NEW] written into every checkpoint. A YOLOv2 checkpoint has a
# completely different detection head, so loading one here should fail with a
# sentence, not with 40 lines of "unexpected key in state_dict".
ARCH = "yolov5"     # >>> [YOLOv5-CHANGED]

# >>> [YOLOv4-NEW] YOLOv4's backbone (CSPDarknet53) uses Mish,
# x * tanh(softplus(x)), instead of LeakyReLU. It is smooth and lets a little
# negative signal through, which the paper found worth ~+1% top-1 on the
# backbone. The neck and the other heads keep LeakyReLU, as in YOLOv4.
#
# >>> [YOLOv5-CHANGED] YOLOv5 uses SiLU, x * sigmoid(x), in the backbone AND
# the neck. Same smooth shape as Mish, cheaper to compute. The segmentation
# head still uses the "leaky" default, so it is untouched.
def activation(name="leaky"):
    if name == "mish":
        return nn.Mish(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    return nn.LeakyReLU(0.1, inplace=True)


def conv_block(channel_in, channel_out, k=3, s=1, act="leaky", p=None):     # >>> [YOLOv4-CHANGED] added `act`, [YOLOv5] `p`
    # s=1 only thickening, s=2 shrinking and thickening
    return nn.Sequential(
        # Converting the picture into thinkened and shrinked picture
        nn.Conv2d(channel_in, channel_out, k, stride=s, padding=(k // 2) if p is None else p, bias=False),
        nn.BatchNorm2d(channel_out),
        activation(act)
    )


# >>> [YOLOv5-CHANGED] YOLOv3's residual block, renamed to what YOLOv5 calls
# it. `shortcut=False` is used in the neck, where the input and output mean
# different things (concatenated features from two scales), so adding them
# would be wrong.
class Bottleneck(nn.Module):
    def __init__(self, channels, shortcut=True, act="silu"):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(channels, channels, k=1, act=act),
            conv_block(channels, channels, k=3, act=act),
        )
        self.shortcut = shortcut

    def forward(self, x):
        y = self.net(x)
        return x + y if self.shortcut else y


class C3(nn.Module):
    # >>> [YOLOv5-NEW] YOLOv5's CSP block, replacing YOLOv4's CSPStage.
    #
    #   x --1x1--> n Bottlenecks --+
    #                              concat --1x1--> out
    #   x --1x1--------------------+
    #
    # Same cross-stage idea as YOLOv4 (half the channels skip the blocks), with
    # one fewer conv: YOLOv4 put an extra 1x1 after the blocks, YOLOv5 found
    # it was not needed. Used in the backbone (with shortcuts) AND the neck
    # (without), so the whole network is built from one block type.
    def __init__(self, channel_in, channel_out, n=1, shortcut=True, act="silu"):
        super().__init__()
        half = channel_out // 2
        self.cv1 = conv_block(channel_in, half, k=1, act=act)
        self.cv2 = conv_block(channel_in, half, k=1, act=act)
        self.m = nn.Sequential(*[Bottleneck(half, shortcut, act) for _ in range(n)])
        self.cv3 = conv_block(2 * half, channel_out, k=1, act=act)

    def forward(self, x):
        return self.cv3(th.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))


class SPPF(nn.Module):
    # >>> [YOLOv5-NEW] "SPP-Fast", replacing YOLOv4's SPP. Three 5x5 max-pools
    # in a row see 5x5, 9x9 and 13x13 windows -- mathematically the same
    # output as SPP(5, 9, 13), because max-pooling a max-pool widens the
    # window -- but each pool reuses the previous one's work, so it is ~2x
    # faster. A 1x1 halves the channels first to keep the concat small.
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


def stage(channel_in, channel_out, n_blocks):
    # >>> [YOLOv5-CHANGED] stride-2 conv + C3 (was CSPStage + Mish in YOLOv4)
    return nn.Sequential(conv_block(channel_in, channel_out, s=2, act="silu"),
                         C3(channel_out, channel_out, n_blocks))


class Backbone(nn.Module):
    # >>> [YOLOv5-CHANGED] CSPDarknet with C3 blocks + SiLU (YOLOv5 v6+).
    # Stem: ONE 6x6 stride-2 conv (padding 2), which is what YOLOv5 v6.0
    # replaced its old "Focus" slice-and-concat layer with -- same output
    # size, same receptive field, but a plain conv that exports and runs
    # faster everywhere. Then 5 stages of stride-2 conv + C3.
    # depths (1, 2, 3, 3, 1) follows YOLOv5's 3 : 6 : 9 : 3 stage pattern at
    # small-model depth. Resolutions / channels / outputs are unchanged, so
    # the segmentation and classification heads see the same tensors as before.
    def __init__(self, in_channels=4, depths=(1, 2, 3, 3, 1)):
        super().__init__()
        # Reducing the pixels into 7x7 (448 / 2^6)
        # 7x7 being the box of determining the size of the grid cell
        self.s1 = conv_block(in_channels, 32, k=6, s=2, p=2, act="silu")
        self.s2 = stage(32, 64, depths[0])
        self.s3 = stage(64, 128, depths[1])
        self.s4 = stage(128, 256, depths[2])
        self.s5 = stage(256, 512, depths[3])
        self.s6 = stage(512, 1024, depths[4])


    def forward(self, x):
        # Forward pass to get data from size 7 to 56 and later hand to SegmentationHead to help to classify the border
        # Mainly using s3 to s5 to calibrate the shape, therefore keeping the record all these data and ready to share backbone
        f56 = self.s3(self.s2(self.s1(x)))
        f28 = self.s4(f56)
        f14 = self.s5(f28)
        f7 = self.s6(f14)
        return f56, f28, f14, f7


class PANNeck(nn.Module):
    # >>> [YOLOv5-CHANGED] same PANet shape as YOLOv4 (top-down, then
    # bottom-up, merged by concatenation) but built from C3 blocks with SiLU
    # instead of plain conv_sets, and with SPPF instead of SPP on top.
    # The neck also no longer narrows the channels as it goes up: the outputs
    # are 256 / 512 / 1024 for 28x28 / 14x14 / 7x7, like YOLOv5's P3 / P4 / P5.
    #
    #   f7 -> SPPF -> 1x1 (h7) -----------------------------------> concat -> C3 -> n7   (7x7 head)
    #                  |                                              ^
    #                  up x2                                          3x3 s2
    #                  v                                              |
    #   f14 ------> concat -> C3 -> 1x1 (h14) ------------------> concat -> C3 -> n14  (14x14 head)
    #                                |                    ^
    #                                up x2                3x3 s2
    #                                v                    |
    #   f28 ----------------------> concat -> C3 -> n28 --+                          (28x28 head)
    def __init__(self, n=1, act="silu"):
        super().__init__()
        self.sppf = SPPF(1024, 1024, act=act)
        self.h7 = conv_block(1024, 512, k=1, act=act)
        self.td14 = C3(512 + 512, 512, n, shortcut=False, act=act)
        self.h14 = conv_block(512, 256, k=1, act=act)
        self.td28 = C3(256 + 256, 256, n, shortcut=False, act=act)
        self.down28 = conv_block(256, 256, s=2, act=act)
        self.bu14 = C3(256 + 256, 512, n, shortcut=False, act=act)
        self.down14 = conv_block(512, 512, s=2, act=act)
        self.bu7 = C3(512 + 512, 1024, n, shortcut=False, act=act)
        self.out_channels = (1024, 512, 256)    # n7, n14, n28

    def forward(self, f28, f14, f7):
        h7 = self.h7(self.sppf(f7))
        h14 = self.h14(self.td14(th.cat([F.interpolate(h7, scale_factor=2), f14], dim=1)))
        n28 = self.td28(th.cat([F.interpolate(h14, scale_factor=2), f28], dim=1))
        n14 = self.bu14(th.cat([self.down28(n28), h14], dim=1))
        n7 = self.bu7(th.cat([self.down14(n14), h7], dim=1))
        return n7, n14, n28


class DetectionHead(nn.Module):
    # >>> [YOLOv2-CHANGED] Was: B free boxes per cell, raw regression (YOLOv1).
    # Now: one prediction per ANCHOR, and the raw numbers are read as
    # CORRECTIONS to that anchor rather than as the box itself.
    #   bx = (sigmoid(tx) + col) / S      centre is locked inside its own cell
    #   by = (sigmoid(ty) + row) / S
    #   bw = anchor_w * exp(tw)           width is a multiple of the prior
    #   bh = anchor_h * exp(th)
    #   conf = sigmoid(to)                confidence is forced into [0, 1]
    # This is YOLOv2's "direct location prediction". YOLOv1 let the network
    # output any number for cx, so early in training a box could be predicted
    # on the other side of the image and the model took many epochs to settle.
    # The sigmoid makes that impossible by construction.
    #
    # >>> [YOLOv3-CHANGED] Was: one 7x7 grid with every anchor on it.
    # Now: three grids (GRIDS = 7, 14, 28) fed by the FPN neck, and each
    # anchor lives on exactly ONE of them (split_anchors_by_scale). The decode
    # formula above is unchanged, it is just applied per grid with that grid's
    # S. All boxes from all grids are then flattened into one [N, M, 5]
    # tensor of (cx, cy, w, h, conf) in [0, 1], so evaluate.py, visualize.py
    # and the web demo still just pick the most confident row.
    #
    # >>> [YOLOv4-CHANGED] "eliminate grid sensitivity": stretch the sigmoid a
    # little past [0, 1] so a centre on a cell border is reachable.
    #
    # >>> [YOLOv5-CHANGED] new decode:
    #   bx = (2 * sigmoid(tx) - 0.5 + col) / S     centre in (col - 0.5, col + 1.5)
    #   bw = anchor_w * (2 * sigmoid(tw)) ** 2     width in (0, 4 * anchor_w)
    # - The centre range is wide enough that a cell can also predict a hand
    #   centred in the NEIGHBOURING cell, which the loss now uses to give
    #   every hand three positive cells instead of one (see build_targets).
    # - exp(tw) is gone. exp is unbounded, which is why every earlier branch
    #   had to bound tw to +-4 (hard clamp on v2, 4 * tanh(tw / 4) since v3);
    #   (2 * sigmoid)^2 is bounded by construction at 4x the anchor --
    #   matching the anchor_t = 4 rule the loss uses to decide which anchors
    #   may predict a box at all.
    def __init__(self, B=2, anchors=None, in_channels=(1024, 512, 256), grids=GRIDS):
        super().__init__()
        # `B` is kept only so old call sites do not crash; the real count is
        # the number of anchors.
        self.anchors_list = [tuple(a) for a in (ANCHORS if anchors is None else anchors)]
        self.grids = tuple(grids)
        self.groups = split_anchors_by_scale(self.anchors_list, len(self.grids))
        self.B = len(self.anchors_list)
        # register_buffer: moves with .to(device) and is saved in the
        # checkpoint, so an evaluation run can never use different anchors
        # from the run that trained the weights.
        # >>> [YOLOv3-CHANGED] stored in GROUPED order (coarse grid first),
        # which is the order the boxes come out in.
        self.register_buffer("anchors", anchor_tensor([a for g in self.groups for a in g]))

        # >>> [YOLOv5-CHANGED] one 1x1 conv per scale emitting (tx, ty, tw, th,
        # to) for each anchor (YOLOv5's Detect layer). YOLOv3/4 had an extra 3x3
        # here; YOLOv5's C3 neck already ends in a 3x3, so it was redundant.
        self.preds = nn.ModuleList()
        for c, group in zip(in_channels, self.groups):
            self.preds.append(nn.Sequential(nn.Conv2d(c, len(group) * 5, kernel_size=1)))
        self._init_objectness_bias()

        # --- per-anchor bookkeeping the loss needs ---------------------------
        # For anchor a (in grouped order): the grid it lives on, where its
        # scale starts in the flattened output, its index inside that scale and
        # how many anchors share the scale. Flat index of (anchor, row, col):
        #   offset[a] + (row * S[a] + col) * count[a] + local[a]
        grid_of, offset_of, local_of, count_of = [], [], [], []
        offset = 0
        for S_i, group in zip(self.grids, self.groups):
            for j in range(len(group)):
                grid_of.append(S_i)
                offset_of.append(offset)
                local_of.append(j)
                count_of.append(len(group))
            offset += S_i * S_i * len(group)
        self.n_boxes = offset
        # persistent=False: derived from the anchors, so there is nothing to
        # save and old/new checkpoints stay key-compatible.
        self.register_buffer("anchor_grid", th.tensor(grid_of), persistent=False)
        self.register_buffer("anchor_offset", th.tensor(offset_of), persistent=False)
        self.register_buffer("anchor_local", th.tensor(local_of), persistent=False)
        self.register_buffer("anchor_count", th.tensor(count_of), persistent=False)

    def _init_objectness_bias(self):
        # >>> [YOLOv3-NEW] start every objectness at p = 0.01 instead of 0.5.
        # There are ~2000 boxes per image and at most one hand. At p = 0.5 the
        # ~2000 "no object" BCE terms add up to a loss in the hundreds on step
        # one, and since the backbone is shared that spike also hits the
        # classification and segmentation heads. Starting near the true
        # background rate removes it. (Trick from RetinaNet's focal-loss
        # paper; YOLOv5 does the same.)
        #
        # >>> [YOLOv5-CHANGED] the prior now depends on the grid, as in YOLOv5
        # (which uses log(8 / cells) for ~8 objects per COCO image): here there
        # is one hand per image, so the prior is 1 / S^2 -- 1/49 on the 7x7 grid
        # and 1/784 on the 28x28 one.
        for pred, S in zip(self.preds, self.grids):
            out = pred[-1]
            with th.no_grad():
                out.bias.view(-1, 5)[:, 4].fill_(math.log(1.0 / S ** 2))

    def layout(self):
        """Everything yolo_detection_loss needs to map a box to its flat index."""
        # >>> [YOLOv5-NEW] "scales": (S, first flat index, last + 1) per grid, for
        # the per-scale objectness balance in the loss.
        scales, start = [], 0
        for S_i, group in zip(self.grids, self.groups):
            end = start + S_i * S_i * len(group)
            scales.append((S_i, start, end))
            start = end
        return {"anchors": self.anchors, "grid": self.anchor_grid,
                "offset": self.anchor_offset, "local": self.anchor_local,
                "count": self.anchor_count, "scales": scales}

    def _decode(self, t, S, anchors):
        # t: [N, S, S, A, 5] raw. Same YOLOv2 formula as before, per grid.
        # .float(): decode in fp32 even under autocast, so sigmoid/exp do not
        # saturate to exactly 0 or 1 in half precision.
        t = t.float()
        A = anchors.size(0)
        cols = th.arange(S, device=t.device, dtype=t.dtype).view(1, 1, S, 1)
        rows = th.arange(S, device=t.device, dtype=t.dtype).view(1, S, 1, 1)
        # >>> [YOLOv5-CHANGED] see the class comment
        y = th.sigmoid(t)
        cx = (y[..., 0] * 2 - 0.5 + cols) / S
        cy = (y[..., 1] * 2 - 0.5 + rows) / S
        aw = anchors[:, 0].view(1, 1, 1, A).to(t.dtype)
        ah = anchors[:, 1].view(1, 1, 1, A).to(t.dtype)
        w = aw * (y[..., 2] * 2) ** 2
        h = ah * (y[..., 3] * 2) ** 2
        conf = y[..., 4]
        return th.stack([cx, cy, w, h, conf], dim=-1)

    def forward(self, features):
        # features: (p7, p14, p28) from the neck, coarse first like GRIDS
        outs, start = [], 0
        for x, pred, S, group in zip(features, self.preds, self.grids, self.groups):
            A = len(group)
            assert x.shape[-1] == S, f"expected a {S}x{S} map, got {tuple(x.shape)}"
            # Convert back to the form which torch expects (batch, height, width, channel)
            t = pred(x).permute(0, 2, 3, 1).reshape(x.size(0), S, S, A, 5)
            boxes = self._decode(t, S, self.anchors[start:start + A])
            outs.append(boxes.reshape(x.size(0), -1, 5))
            start += A
        # [N, M, 5], M = 7*7*A7 + 14*14*A14 + 28*28*A28
        return th.cat(outs, dim=1)


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
    def __init__(self, in_channels=5, n_classes=10, B=2, anchors=None):   # >>> [YOLOv2-CHANGED] added `anchors`
        super().__init__()
        self.backbone = Backbone(in_channels)
        # >>> [YOLOv3-NEW] the neck only feeds detection. Segmentation keeps its
        # own U-Net decoder and classification keeps reading f7, so neither of
        # those heads changes behaviour because of the new detector.
        self.neck = PANNeck()     # >>> [YOLOv4-CHANGED] was FPNNeck
        self.detecting_head = DetectionHead(B, anchors, in_channels=self.neck.out_channels)   # >>> [YOLOv3-CHANGED]
        # >>> [YOLOv2-NEW] convenience handle for train.py / evaluate.py
        self.anchors_list = self.detecting_head.anchors_list
        self.classification_head = ClassificationHead(n_classes)
        self.segmentation_head = SegmentationHead()

    
    def forward(self, x):
        f56, f28, f14, f7 = self.backbone(x)
        return (self.detecting_head(self.neck(f28, f14, f7)),    # >>> [YOLOv3-CHANGED]
                self.segmentation_head(f56, f28, f14, f7),
                self.classification_head(f7)
                )


# >>> [YOLOv2-NEW]
def load_handgesture_net(checkpoint, in_channels=4, n_classes=10, device=None):
    """
    Rebuild the network with the SAME anchors it was trained with, then pour
    the weights in.

    Why this has to exist: the weights of the detection head are corrections
    relative to specific anchor shapes. Loading them into a model that was
    built with different anchors gives you a model that runs, produces boxes,
    and is quietly wrong. So the anchors are read back out of the checkpoint,
    never re-guessed.
    """
    # >>> [YOLOv3-NEW] refuse checkpoints from another architecture up front
    arch = checkpoint.get("arch")
    if arch != ARCH:
        raise RuntimeError(
            f"This checkpoint was trained with arch={arch!r} (None = YOLOv2 or "
            f"older) but this branch builds {ARCH!r}. The detection heads are "
            f"different networks; evaluate it on the branch it was trained on, "
            f"or retrain here.")
    anchors = checkpoint.get("anchors")
    if anchors is None:
        state = checkpoint.get("model_state", {})
        if "detecting_head.anchors" in state:
            anchors = state["detecting_head.anchors"].tolist()
        else:
            raise RuntimeError(
                "This checkpoint stores no anchors, so it comes from the "
                "pre-YOLOv2 code. Evaluate it on the branch it was trained on, "
                "or retrain on this branch.")
    model = HandGestureNet(in_channels=in_channels, n_classes=n_classes,
                           anchors=anchors)
    model.load_state_dict(checkpoint["model_state"])
    if device is not None:
        model = model.to(device)
    return model
