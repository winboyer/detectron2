from .cascade_mask_rcnn_mvitv2_t_3x_smoke import model, dataloader, optimizer, lr_multiplier, train


model.backbone.bottom_up.depth = 24
model.backbone.bottom_up.last_block_indexes = (1, 4, 20, 23)
model.backbone.bottom_up.drop_path_rate = 0.4
model.backbone.bottom_up.frozen_stages = 24

# train.init_checkpoint = "detectron2://ImageNetPretrained/mvitv2/MViTv2_B_in1k.pyth"
train.init_checkpoint = "/root/jinyfeng/models/vitdet-models/model_final_cascade_mask_rcnn_mvitv2_b_3x.pkl"
train.output_dir = 'smoke_cas_mvitv2_b_model_50ep'
