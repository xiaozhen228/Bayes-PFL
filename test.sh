
# test on the MVTec-AD dataset
(
nohup python -u test.py --dataset mvtec --data_path ./dataset/mvisa/data \
--checkpoint_path ./my_exps/train_visa/XXXXXX \
--save_path ./results/test_mvtec --pretrained_path ./pretrained_weight/ViT-L-14-336px.pt \
--prompt_len 2 --deep_prompt_len 1 --device_id 0 --features_list 6 12 18 24 --pretrained openai --image_size 518 \
--seed 333 --config_path ./models/model_configs/ViT-L-14-336.json --model ViT-L-14-336
) > ./log_test_mvtec.out 2>&1 &

# test on the VisA dataset
(
nohup python -u test.py --dataset visa --data_path ./dataset/mvisa/data \
--checkpoint_path ./my_exps/train_mvtec/XXXXXX \
--save_path ./results/test_visa --pretrained_path ./pretrained_weight/ViT-L-14-336px.pt \
--prompt_len 2 --deep_prompt_len 1 --device_id 0 --features_list 6 12 18 24 --pretrained openai --image_size 518 \
--seed 333 --config_path ./models/model_configs/ViT-L-14-336.json --model ViT-L-14-336
) > ./log_test_visa.out 2>&1 &