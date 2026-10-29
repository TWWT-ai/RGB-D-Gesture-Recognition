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
ARCH = "yolov4"     # >>> [YOLOv4-CHANGED]

# >>> [YOLOv4-NEW] YOLOv4's backbone (CSPDarknet53) uses Mish,
# x * tanh(softplus(x)), instead of LeakyReLU. It is smooth and lets a little
# negative signal through, which the paper found worth ~+1% top-1 on the
# backbone. The neck and the other heads keep LeakyReLU, as in YOLOv4.
def activation(name="leaky"):
    if name == "mish":
        return nn.Mish(inplace=True)
    return nn.LeakyReLU(0.1, inplace=True)


def conv_block(channel_in, channel_out, k=3, s=1, act="leaky"):     # >>> [YOLOv4-CHANGED] added `act`
    # s=1 only thickening, s=2 shrinking and thickening
    return nn.Sequential(
        # Converting the picture into thinkened and shrinked picture
        nn.Conv2d(channel_in, channel_out, k, stride=s, padding=(k // 2), bias=False),
        nn.BatchNorm2d(channel_out),
        activation(act)
    )


# >>> [YOLOv3-NEW] Darknet-53's building block.
class ResidualBlock(nn.Module):
    # 1x1 squeezes the channels to half, 3x3 looks at the neighbourhood and
    # restores them, then the input is ADDED back. The shortcut gives the
    # gradient a straight road back to the early layers, which is what let
    # YOLOv3 go from Darknet-19 (no shortcuts) to a 53-layer backbone.
    # >>> [YOLOv4-CHANGED] `hidden` and `act` added: inside a CSP stage the
    # block does not squeeze (hidden = channels), and the backbone uses Mish.
    def __init__(self, channels, hidden=None, act="leaky"):
        super().__init__()
        hidden = channels // 2 if hidden is None else hidden
        self.net = nn.Sequential(
            conv_block(channels, hidden, k=1, act=act),
            conv_block(hidden, channels, k=3, act=act),
        )

    def forward(self, x):
        return x + self.net(x)


class CSPStage(nn.Module):
    # >>> [YOLOv4-NEW] Cross Stage Partial stage (CSPNet), replacing YOLOv3's
    # plain Darknet stage.
    #
    #   x --stride-2 conv--+--1x1--> residual blocks --1x1--+
    #                      |                                concat --1x1--> out
    #                      +--1x1-------------------------- +
    #
    # Only HALF the channels go through the residual blocks; the other half
    # skips the whole stage and is concatenated back at the end. The blocks
    # therefore cost about half as much, and the gradient reaching the early
    # layers comes through two different routes, so the stage does not keep
    # learning the same thing twice. CSPNet reports similar or better accuracy
    # at noticeably lower compute.
    def __init__(self, channel_in, channel_out, n_blocks, act="mish"):
        super().__init__()
        half = channel_out // 2
        self.down = conv_block(channel_in, channel_out, s=2, act=act)
        self.part1 = conv_block(channel_out, half, k=1, act=act)       # cross-stage route
        self.part2 = nn.Sequential(
            conv_block(channel_out, half, k=1, act=act),
            *[ResidualBlock(half, hidden=half, act=act) for _ in range(n_blocks)],
            conv_block(half, half, k=1, act=act),
        )
        self.transition = conv_block(2 * half, channel_out, k=1, act=act)

    def forward(self, x):
        x = self.down(x)
        return self.transition(th.cat([self.part2(x), self.part1(x)], dim=1))


def stage(channel_in, channel_out, n_blocks):
    # >>> [YOLOv4-CHANGED] was a plain Darknet-53 stage (YOLOv3); now CSP + Mish
    return CSPStage(channel_in, channel_out, n_blocks, act="mish")


class Backbone(nn.Module):
    # >>> [YOLOv4-CHANGED] each stage is now a CSPStage with Mish (CSPDarknet53).
    # Resolutions / channels / outputs are still identical, so nothing that
    # reads f56..f7 has to change.
    # >>> [YOLOv3-CHANGED] Was: each stage = stride-2 conv + one plain 3x3 conv.
    # Now: each stage = stride-2 conv + `depths[i]` residual blocks
    # (Darknet-53 style). Resolutions and channel counts are UNCHANGED, so the
    # SegmentationHead and ClassificationHead see exactly the tensors they saw
    # before.
    # Darknet-53 uses (1, 2, 8, 8, 4) after its stem on ImageNet-sized data;
    # this network is trained from scratch on a few thousand frames, so the
    # default is a shallower (1, 1, 2, 4, 4, 2). Pass depths=... to go deeper.
    def __init__(self, in_channels=4, depths=(1, 1, 2, 4, 4, 2)):
        super().__init__()
        # Reducing the pixels into 7x7 (448 / 2^6)
        # 7x7 being the box of determining the size of the grid cell
        self.s1 = stage(in_channels, 32, depths[0])
        self.s2 = stage(32, 64, depths[1])
        self.s3 = stage(64, 128, depths[2])
        self.s4 = stage(128, 256, depths[3])
        self.s5 = stage(256, 512, depths[4])
        self.s6 = stage(512, 1024, depths[5])


    def forward(self, x):
        # Forward pass to get data from size 7 to 56 and later hand to SegmentationHead to help to classify the border
        # Mainly using s3 to s5 to calibrate the shape, therefore keeping the record all these data and ready to share backbone
        f56 = self.s3(self.s2(self.s1(x)))
        f28 = self.s4(f56)
        f14 = self.s5(f28)
        f7 = self.s6(f14)
        return f56, f28, f14, f7


# >>> [YOLOv3-NEW]
def conv_set(channel_in, channel_mid):
    # Alternating 1x1 / 3x3 / 1x1. YOLOv3 uses five of these in a row; three
    # keeps the neck at a size a few thousand training frames can support.
    return nn.Sequential(
        conv_block(channel_in, channel_mid, k=1),
        conv_block(channel_mid, channel_mid * 2, k=3),
        conv_block(channel_mid * 2, channel_mid, k=1),
    )


class SPP(nn.Module):
    # >>> [YOLOv4-NEW] Spatial Pyramid Pooling block (YOLOv4's SPP, from
    # YOLOv3-SPP). Max-pool the deepest map with 5x5, 9x9 and 13x13 windows at
    # stride 1 and stack the results with the input. On a 7x7 map a 13x13
    # window covers the whole image, so every cell of the 7x7 grid now also
    # sees a global summary of the frame, at almost zero extra cost.
    def __init__(self, pools=(5, 9, 13)):
        super().__init__()
        self.pools = nn.ModuleList(nn.MaxPool2d(k, stride=1, padding=k // 2) for k in pools)

    def forward(self, x):
        return th.cat([x] + [pool(x) for pool in self.pools], dim=1)


class PANNeck(nn.Module):
    # >>> [YOLOv4-NEW] replaces YOLOv3's FPNNeck: SPP on top, then PANet.
    #
    # YOLOv3's FPN only goes top-down, so the 7x7 head never sees the precise
    # low-level detail from f28. PANet adds a second, bottom-up path after the
    # top-down one, so localisation detail flows back up to the coarse grid
    # through only a couple of layers. YOLOv4 merges paths by CONCATENATION
    # (its "modified PAN") instead of PANet's addition.
    #
    #   f7 -> conv_set -> SPP -> conv_set -> p7 -----------------------> concat -> conv_set -> n7  (7x7 head)
    #                                        |                            ^
    #                                        1x1, up x2                   stride-2 conv
    #                                        v                            |
    #   f14 -> 1x1 --------------------> concat -> conv_set -> t14 ----> concat -> conv_set -> n14 (14x14 head)
    #                                                          |          ^
    #                                                          1x1, up x2 stride-2 conv
    #                                                          v          |
    #   f28 -> 1x1 --------------------------------------> concat -> conv_set -> n28              (28x28 head)
    def __init__(self):
        super().__init__()
        # top of the pyramid + SPP
        self.pre7 = conv_set(1024, 512)
        self.spp = SPP()
        self.post7 = conv_set(512 * 4, 512)
        # top-down
        self.lat7 = conv_block(512, 256, k=1)
        self.in14 = conv_block(512, 256, k=1)
        self.td14 = conv_set(256 + 256, 256)
        self.lat14 = conv_block(256, 128, k=1)
        self.in28 = conv_block(256, 128, k=1)
        self.td28 = conv_set(128 + 128, 128)
        # bottom-up
        self.down28 = conv_block(128, 256, s=2)
        self.bu14 = conv_set(256 + 256, 256)
        self.down14 = conv_block(256, 512, s=2)
        self.bu7 = conv_set(512 + 512, 512)
        self.out_channels = (512, 256, 128)    # n7, n14, n28

    def forward(self, f28, f14, f7):
        p7 = self.post7(self.spp(self.pre7(f7)))
        t14 = self.td14(th.cat([F.interpolate(self.lat7(p7), scale_factor=2), self.in14(f14)], dim=1))
        n28 = self.td28(th.cat([F.interpolate(self.lat14(t14), scale_factor=2), self.in28(f28)], dim=1))
        n14 = self.bu14(th.cat([self.down28(n28), t14], dim=1))
        n7 = self.bu7(th.cat([self.down14(n14), p7], dim=1))
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
    # >>> [YOLOv4-CHANGED] "eliminate grid sensitivity":
    #   bx = (scale_xy * sigmoid(tx) - (scale_xy - 1) / 2 + col) / S
    # With a plain sigmoid, a hand whose centre sits exactly on a cell border
    # needs sigmoid(tx) = 0 or 1, i.e. tx = -inf / +inf, which the network can
    # never output. Stretching the sigmoid slightly past [0, 1] makes the
    # border reachable with finite tx. YOLOv4's configs use 1.2 / 1.1 / 1.05
    # from the finest to the coarsest grid (finer grid = more borders).
    SCALE_XY = {7: 1.05, 14: 1.1, 28: 1.2}

    def __init__(self, B=2, anchors=None, in_channels=(512, 256, 128), grids=GRIDS):
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

        # One small predictor per scale: 3x3 to look around, 1x1 to emit
        # (tx, ty, tw, th, to) for each of that scale's anchors.
        self.preds = nn.ModuleList()
        for c, group in zip(in_channels, self.groups):
            out = nn.Conv2d(c * 2, len(group) * 5, kernel_size=1)
            self.preds.append(nn.Sequential(conv_block(c, c * 2), out))
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

    def _init_objectness_bias(self, prior=0.01):
        # >>> [YOLOv3-NEW] start every objectness at p = 0.01 instead of 0.5.
        # There are ~2000 boxes per image and at most one hand. At p = 0.5 the
        # ~2000 "no object" BCE terms add up to a loss in the hundreds on step
        # one, and since the backbone is shared that spike also hits the
        # classification and segmentation heads. Starting near the true
        # background rate removes it. (Trick from RetinaNet's focal-loss
        # paper; YOLOv5 does the same.)
        for pred in self.preds:
            out = pred[-1]
            with th.no_grad():
                out.bias.view(-1, 5)[:, 4].fill_(math.log(prior / (1 - prior)))

    def layout(self):
        """Everything yolo_detection_loss needs to map a box to its flat index."""
        return {"anchors": self.anchors, "grid": self.anchor_grid,
                "offset": self.anchor_offset, "local": self.anchor_local,
                "count": self.anchor_count}

    def _decode(self, t, S, anchors):
        # t: [N, S, S, A, 5] raw. Same YOLOv2 formula as before, per grid.
        # .float(): decode in fp32 even under autocast, so sigmoid/exp do not
        # saturate to exactly 0 or 1 in half precision.
        t = t.float()
        A = anchors.size(0)
        cols = th.arange(S, device=t.device, dtype=t.dtype).view(1, 1, S, 1)
        rows = th.arange(S, device=t.device, dtype=t.dtype).view(1, S, 1, 1)
        sxy = self.SCALE_XY.get(S, 1.0)     # >>> [YOLOv4-CHANGED]
        cx = (sxy * th.sigmoid(t[..., 0]) - (sxy - 1) / 2 + cols) / S
        cy = (sxy * th.sigmoid(t[..., 1]) - (sxy - 1) / 2 + rows) / S
        # clamp before exp: an untrained network can emit large values and
        # exp() would overflow to inf, which poisons the whole loss.
        aw = anchors[:, 0].view(1, 1, 1, A).to(t.dtype)
        ah = anchors[:, 1].view(1, 1, 1, A).to(t.dtype)
        w = aw * th.exp(t[..., 2].clamp(-4.0, 4.0))
        h = ah * th.exp(t[..., 3].clamp(-4.0, 4.0))
        conf = th.sigmoid(t[..., 4])
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
