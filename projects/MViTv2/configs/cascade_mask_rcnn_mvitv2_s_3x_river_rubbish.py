from .cascade_mask_rcnn_mvitv2_t_3x_river_rubbish import model, dataloader, optimizer, lr_multiplier, train


model.backbone.bottom_up.depth = 16
model.backbone.bottom_up.last_block_indexes = (0, 2, 13, 15)
model.backbone.bottom_up.frozen_stages = 16

# train.init_checkpoint = "detectron2://ImageNetPretrained/mvitv2/MViTv2_S_in1k.pyth"
train.init_checkpoint = "/root/jinyfeng/models/vitdet-models/model_final_cascade_mask_rcnn_mvitv2_s_3x.pkl"
train.output_dir = 'river_rubbish_cas_mvitv2_s_model_50ep'
