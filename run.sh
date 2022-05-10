
python turngpt/train_conditional.py   --gpus -1 \
                        --batch_size 2 \
                        --accumulate_grad_batches 10\
                        --trp_projection_steps 1\
                        --strategy=ddp

# export CUDA_VISIBLE_DEVICES=7
# python turngpt/model.py

# export CUDA_VISIBLE_DEVICES=5
# python turngpt/train.py   --gpus 1 \
#                         --batch_size 4 \
#                         --accumulate_grad_batches 10\
#                         --trp_projection_steps 1