
#mvitv2-t
python ./tools/lazyconfig_train_net.py --config-file projects/MViTv2/configs/mask_rcnn_mvitv2_t_3x_cookhat.py > cookhat_mvitv2-t_50ep.log

#cas_mvitv2-t
python ./tools/lazyconfig_train_net.py --config-file projects/MViTv2/configs/cascade_mask_rcnn_mvitv2_t_3x_cookhat.py > cookhat_cas_mvitv2-t_50ep.log

#cas_mvitv2-s
python ./tools/lazyconfig_train_net.py --config-file projects/MViTv2/configs/cascade_mask_rcnn_mvitv2_s_3x_cookhat.py > cookhat_cas_mvitv2-s_50ep.log

#cas_mvitv2-b
python ./tools/lazyconfig_train_net.py --config-file projects/MViTv2/configs/cascade_mask_rcnn_mvitv2_b_3x_cookhat.py > cookhat_cas_mvitv2-b_50ep.log

#cas_mvitv2-b_in21k
python ./tools/lazyconfig_train_net.py --config-file projects/MViTv2/configs/cascade_mask_rcnn_mvitv2_b_in21k_3x_cookhat.py > cookhat_cas_mvitv2-b_in21k_50ep.log

#vitdet, vit-b_IN1k_MAE, model_final_vit-b_IN1K_MAE.pkl
python ./tools/lazyconfig_train_net.py --config-file projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep_cookhat.py > cookhat_vitdet-b_100ep.log

#vitdet, vit-b_IN1k_MAE_cas, model_final_vit-b_IN1K_MAE_cas.pkl
python ./tools/lazyconfig_train_net.py --config-file projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_b_100ep_cookhat.py > cookhat_cas_vitdet-b_100ep.log

#vitdet, cas_mvitv2-b_IN21K_SUP, model_final_mvitv2-b_IN21K_SUP.pkl
python ./tools/lazyconfig_train_net.py --config-file projects/ViTDet/configs/COCO/cascade_mask_rcnn_mvitv2_b_in21k_100ep_cookhat.py > cookhat_cas_mvitv2_b_in21k_v2_model_100ep_v2.log

#vitdet, cas_swin-b_IN21K_SUP, model_final_swin-b_IN21K_SUP.pkl
python ./tools/lazyconfig_train_net.py --config-file projects/ViTDet/configs/COCO/cascade_mask_rcnn_swin_b_in21k_50ep_cookhat.py > cookhat_cas_swin_b_in21k_model_50ep.log



