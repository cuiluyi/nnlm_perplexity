export CUDA_VISIBLE_DEVICES=3
python3 test.py --recipe recipes/demo.yaml > logs/test_$(date +"%Y%m%d_%H%M%S").log 2>&1