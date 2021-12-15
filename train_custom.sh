python main.py \
--root_path ./data \
--video_path image_data \
--annotation_path json_file/custom.json \
--result_path results \
--dataset custom \
--n_classes 2 \
--model resnet \
--model_depth 50 \
--n_pretrain_classes 400 \
--pretrain_path models/resnet-50-kinetics.pth \
--ft_begin_module "" \
--sample_size 160 \
--sample_duration 50 \
--sample_t_stride 3 \
--inference_stride 3 \
--train_crop random \
--train_crop_min_ratio 0.75 \
--colorjitter \
--train_t_crop random \
--learning_rate 0.001 \
--momentum 0.9 \
--dampening 0.0 \
--weight_decay 1e-3 \
--mean_dataset 0.5 \
--value_scale 1 \
--lr_scheduler multistep \
--multistep_milestones 30 60 90 120 150 180 210 160 180 200 \
--gamma_milestones 0.6 \
--batch_size 8 \
--inference_batch_size 1 \
--n_epochs 200 \
--inference_crop nocrop \
--n_threads 2 \
--checkpoint 50 \
--tensorboard \
--train_ts_channel_shuffle


# --no_mean_norm
# --nesterov
# --multistep_milestones
# --batchnorm_sync
# --n_val_samples
# --no_max_pool


