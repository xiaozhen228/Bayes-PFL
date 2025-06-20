# This script uses two NVIDIA 3090 GPUs to run two experiments simultaneously. If you only have one 3090, please comment out one of the experiments.

# train on the VisA dataset
(
nohup python -u train.py --dataset visa --train_data_path ./dataset/mvisa/data \
--val_data_path ./dataset/mvisa/data \
--save_path ./my_exps/train_visa --pretrained_path ./pretrained_weight/ViT-L-14-336px.pt \
--prompt_context_len 5 --prompt_num 3 --prompt_state_len 5 --device_id 0 --learning_rate 0.0001 --features_list 6 12 18 24 --pretrained openai --image_size 518 \
--batch_size 32 --epochs 30 --sample_num 5 --num_flows 10 \
--config_path ./open_clip_local/model_configs/ViT-L-14-336.json --model ViT-L-14-336 --seed 333 --alpha 0.6
) > ./log_train_visa.out 2>&1 &


# train on the MVTec-AD dataset
(
nohup python -u train.py --dataset mvtec --train_data_path ./dataset/mvisa/data \
--val_data_path ./dataset/mvisa/data \
--save_path ./my_exps/train_mvtec --pretrained_path ./pretrained_weight/ViT-L-14-336px.pt \
--prompt_context_len 5 --prompt_num 3 --prompt_state_len 5 --device_id 1 --learning_rate 0.0001 --features_list 6 12 18 24 --pretrained openai --image_size 518 \
--batch_size 32 --epochs 30 --sample_num 5 --num_flows 10 \
--config_path ./open_clip_local/model_configs/ViT-L-14-336.json --model ViT-L-14-336 --seed 333 --alpha 0.6
) > ./log_train_mvtec.out 2>&1 &
