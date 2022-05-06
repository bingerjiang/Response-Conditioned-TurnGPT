export CUDA_VISIBLE_DEVICES=7
python turngpt/train.py   --gpus 1 \
                        --batch_size 2 \
                        --accumulate_grad_batches 10

# export CUDA_VISIBLE_DEVICES=7
# python turngpt/model.py