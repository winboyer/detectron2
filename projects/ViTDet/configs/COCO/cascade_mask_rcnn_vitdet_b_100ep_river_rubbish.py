from functools import partial
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import (
    FastRCNNOutputLayers,
    FastRCNNConvFCHead,
    CascadeROIHeads,
)

from .mask_rcnn_vitdet_b_100ep_river_rubbish import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)
from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler

# arguments that don't exist for Cascade R-CNN
[model.roi_heads.pop(k) for k in ["box_head", "box_predictor", "proposal_matcher"]]

model.roi_heads.update(
    _target_=CascadeROIHeads,
    box_heads=[
        L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[256, 256, 256, 256],
            fc_dims=[1024],
            conv_norm="LN",
        )
        for _ in range(3)
    ],
    box_predictors=[
        L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(w1, w1, w2, w2)),
            cls_agnostic_bbox_reg=True,
            num_classes="${...num_classes}",
        )
        for (w1, w2) in [(10, 5), (20, 10), (30, 15)]
    ],
    proposal_matchers=[
        L(Matcher)(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
        for th in [0.5, 0.6, 0.7]
    ],
)

train.init_checkpoint = (
    # "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth?matching_heuristics=True"
    "/root/jinyfeng/models/vitdet-models/model_final_vit-b_IN1K_MAE_cas.pkl"
)
train.output_dir = 'river_rubbish_cas_vitdet_b_model_100ep'

dataloader.train.total_batch_size = 4

# Schedule
# 100 ep
train.max_iter = 15000    # 600 ~= total_batch_size*max_iter/total_epochs

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[12000, 14000],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}

