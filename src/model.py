import torch as th
import torch.nn as nn
import torch.nn.functional as F

# >>> [YOLOv2-NEW] anchor priors live in utils.py so the loss, the head and
# kmeans_anchors.py all read the same numbers.
from utils import ANCHORS, anchor_tensor

def conv_block(channel_in, channel_out, k=3, s=1):
    # s=1 only thickening, s=2 shrinking and thickening
    return nn.Sequential(
        # Converting the picture into thinkened and shrinked picture
        nn.Conv2d(channel_in, channel_out, k, stride=s, padding=(k // 2), bias=False),
        nn.BatchNorm2d(channel_out),
        nn.LeakyReLU(0.1, inplace=True)
    )


class Backbone(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        # Reducing the pixels into 7x7 (448 / 2^6)
        # 7x7 being the box of determining the size of the grid cell
        self.s1 = nn.Sequential(conv_block(in_channels, 32, s=2), conv_block(32, 32))
        self.s2 = nn.Sequential(conv_block(32, 64, s=2), conv_block(64, 64))
        self.s3 = nn.Sequential(conv_block(64, 128, s=2), conv_block(128, 128))
        self.s4 = nn.Sequential(conv_block(128, 256, s=2), conv_block(256, 256))
        self.s5 = nn.Sequential(conv_block(256, 512, s=2), conv_block(512, 512))
        self.s6 = nn.Sequential(conv_block(512, 1024, s=2), conv_block(1024, 1024))


    def forward(self, x):
        # Forward pass to get data from size 7 to 56 and later hand to SegmentationHead to help to classify the border
        # Mainly using s3 to s5 to calibrate the shape, therefore keeping the record all these data and ready to share backbone
        f56 = self.s3(self.s2(self.s1(x)))
        f28 = self.s4(f56)
        f14 = self.s5(f28)
        f7 = self.s6(f14)
        return f56, f28, f14, f7


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
    def __init__(self, B=2, anchors=None):
        super().__init__()
        # `B` is kept only so old call sites do not crash; the real count is
        # the number of anchors.
        self.anchors_list = list(ANCHORS if anchors is None else anchors)
        self.B = len(self.anchors_list)
        # register_buffer: moves with .to(device) and is saved in the
        # checkpoint, so an evaluation run can never use different anchors
        # from the run that trained the weights.
        self.register_buffer("anchors", anchor_tensor(self.anchors_list))
        self.net = nn.Sequential(
            # Reforming 1024 of data into 512 constructed with newly mixed weights with lower weights become negligible
            conv_block(1024, 512),
            # To determine the frame that actually lands on what we want
            # Candidate box(possible box containing the object) * (x, y, w, h, conf) = B * 5
            nn.Conv2d(512, self.B * 5, kernel_size=1)   # >>> [YOLOv2-CHANGED] B -> number of anchors
        )


    def forward(self, f7):
        output = self.net(f7)
        # Convert back to the form which torch expects (batch, channel, height, width), (batch, 7, 7, 10)
        output = output.permute(0, 2, 3, 1)   # Number is the index

        # >>> [YOLOv2-NEW] decode the raw t-values into real boxes here, so
        # everything downstream (loss, evaluate.py, visualize.py, the web demo)
        # keeps receiving the same [N, S, S, B*5] tensor of (cx, cy, w, h, conf)
        # in [0, 1] that it received before. Nothing outside this file has to
        # know that anchors exist.
        N, S, _, _ = output.shape
        t = output.view(N, S, S, self.B, 5)

        # Cell indices: col varies along the last spatial axis, row along the first.
        cols = th.arange(S, device=t.device, dtype=t.dtype).view(1, 1, S, 1)
        rows = th.arange(S, device=t.device, dtype=t.dtype).view(1, S, 1, 1)

        cx = (th.sigmoid(t[..., 0]) + cols) / S
        cy = (th.sigmoid(t[..., 1]) + rows) / S
        # clamp before exp: an untrained network can emit large values and
        # exp() would overflow to inf, which poisons the whole loss.
        aw = self.anchors[:, 0].view(1, 1, 1, self.B)
        ah = self.anchors[:, 1].view(1, 1, 1, self.B)
        w = aw * th.exp(t[..., 2].clamp(-4.0, 4.0))
        h = ah * th.exp(t[..., 3].clamp(-4.0, 4.0))
        conf = th.sigmoid(t[..., 4])

        return th.stack([cx, cy, w, h, conf], dim=-1).view(N, S, S, self.B * 5)


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
        x = F.interpolate(x, size=(448, 448), mode="bilinear", align_corners=False)
        return self.out(x)
    

class HandGestureNet(nn.Module):
    def __init__(self, in_channels=5, n_classes=10, B=2, anchors=None):   # >>> [YOLOv2-CHANGED] added `anchors`
        super().__init__()
        self.backbone = Backbone(in_channels)
        self.detecting_head = DetectionHead(B, anchors)   # >>> [YOLOv2-CHANGED]
        # >>> [YOLOv2-NEW] convenience handle for train.py / evaluate.py
        self.anchors_list = self.detecting_head.anchors_list
        self.classification_head = ClassificationHead(n_classes)
        self.segmentation_head = SegmentationHead()

    
    def forward(self, x):
        f56, f28, f14, f7 = self.backbone(x)
        return (self.detecting_head(f7),
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
