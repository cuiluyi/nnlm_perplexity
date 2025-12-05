export CUDA_VISIBLE_DEVICES=3
python3 train.py --recipe recipes/demo.yaml > logs/train_$(date +"%Y%m%d_%H%M%S").log 2>&1