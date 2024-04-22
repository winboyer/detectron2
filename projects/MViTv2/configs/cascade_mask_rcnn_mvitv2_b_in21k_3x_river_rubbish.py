from .cascade_mask_rcnn_mvitv2_b_3x_river_rubbish import model, dataloader, optimizer, lr_multiplier, train

# train.init_checkpoint = "detectron2://ImageNetPretrained/mvitv2/MViTv2_B_in21k.pyth"
train.init_checkpoint = "/root/jinyfeng/models/vitdet-models/model_final_cascade_mask_rcnn_mvitv2_b_in21k_3x.pkl"
train.output_dir = 'river_rubbish_cas_mvitv2_b_in21k_model_50ep'