from functools import partial
import torch.nn as nn
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling import MViT

# from .common.coco_loader import dataloader
from .common.coco_loader_smoke import dataloader

model = model_zoo.get_config("common/models/mask_rcnn_fpn.py").model
constants = model_zoo.get_config("common/data/constants.py").constants
model.pixel_mean = constants.imagenet_rgb256_mean
model.pixel_std = constants.imagenet_rgb256_std
model.input_format = "RGB"
model.backbone.bottom_up = L(MViT)(
    embed_dim=96,
    depth=10,
    num_heads=1,
    last_block_indexes=(0, 2, 7, 9),
    residual_pooling=True,
    drop_path_rate=0.2,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    out_features=("scale2", "scale3", "scale4", "scale5"),
    frozen_stages=10
)
model.backbone.in_features = "${.bottom_up.out_features}"

model.roi_heads.num_classes=1


# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
# train.init_checkpoint = "detectron2://ImageNetPretrained/mvitv2/MViTv2_T_in1k.pyth"
train.init_checkpoint = "/root/jinyfeng/models/vitdet-models/model_final_mask_rcnn_mvitv2_t_3x.pkl"
train.output_dir = 'smoke_mvitv2_t_model_50ep_v3'

# dataloader.train.total_batch_size = 64
dataloader.train.total_batch_size = 8

# # 36 epochs
# train.max_iter = 14571    # 3238 ~= total_batch_size*max_iter/total_epochs
# train.max_iter = 13500    # 2940 ~= total_batch_size*max_iter/total_epochs
# 50 epochs
train.max_iter = 20000    # 3238 ~= total_batch_size*max_iter/total_epochs
# train.max_iter = 18750    # 2940 ~= total_batch_size*max_iter/total_epochs
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
#         milestones=[12000, 16000, 18000],
        milestones=[9000, 12000, 14000],
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.overrides = {
    "pos_embed": {"weight_decay": 0.0},
    "rel_pos_h": {"weight_decay": 0.0},
    "rel_pos_w": {"weight_decay": 0.0},
}
optimizer.lr = 1.6e-4
# optimizer.lr = 8e-5
