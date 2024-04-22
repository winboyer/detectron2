from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling import SwinTransformer

from ..common.coco_loader_river_rubbish import dataloader
from .cascade_mask_rcnn_mvitv2_b_in21k_100ep_river_rubbish import model

model.backbone.bottom_up = L(SwinTransformer)(
    depths=[2, 2, 18, 2],
    drop_path_rate=0.4,
    embed_dim=128,
    num_heads=[4, 8, 16, 32],
    frozen_stages=5
)
model.backbone.in_features = ("p0", "p1", "p2", "p3")
model.backbone.square_pad = 1024

# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
# train.init_checkpoint = "detectron2://ImageNetPretrained/swin/swin_base_patch4_window7_224_22k.pth"
train.init_checkpoint = "/root/jinyfeng/models/vitdet-models/model_final_swin-b_IN21K_SUP.pkl"
train.output_dir = 'river_rubbish_cas_swin_b_in21k_model_50ep'

# Schedule
dataloader.train.total_batch_size = 8

# 50 epochs
train.max_iter = 7500    # 600 ~= total_batch_size*max_iter/total_epochs

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
#         milestones=[163889, 177546],
        milestones=[6000, 7000],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

# Rescale schedule
train.max_iter = train.max_iter // 2  # 100ep -> 50ep
lr_multiplier.scheduler.milestones = [
    milestone // 2 for milestone in lr_multiplier.scheduler.milestones
]
lr_multiplier.scheduler.num_updates = train.max_iter


optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.lr = 4e-5
optimizer.weight_decay = 0.05
optimizer.params.overrides = {"relative_position_bias_table": {"weight_decay": 0.0}}
