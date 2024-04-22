from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

from ..common.coco_loader_playphone import dataloader


model = model_zoo.get_config("common/models/mask_rcnn_vitdet_playphone.py").model

# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = (
    # "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth?matching_heuristics=True"
    "/root/jinyfeng/models/vitdet-models/model_final_vit-b_IN1K_MAE.pkl"
)
train.output_dir = 'playphone_vitdet_b_model_100ep'

#dataloader.train.total_batch_size = 64
dataloader.train.total_batch_size = 8

# Schedule
# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
train.max_iter = 32000    # 2460 ~= total_batch_size*max_iter/total_epochs


lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
#         milestones=[163889, 177546],
        milestones=[24000, 28000],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
