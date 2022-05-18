# export CUDA_VISIBLE_DEVICES=7
# python turngpt/train_conditional.py   --gpus 1 \
#                         --batch_size 2 \
#                         --accumulate_grad_batches 10\
#                         --trp_projection_steps 1\
#                         --fast_dev_run

# export CUDA_VISIBLE_DEVICES=7
# python turngpt/model.py

# export CUDA_VISIBLE_DEVICES=5
# python turngpt/train.py   --gpus 1 \
#                         --batch_size 4 \
#                         --accumulate_grad_batches 10\
#                         --trp_projection_steps 1


                       # --fast_dev_run
#export CUDA_VISIBLE_DEVICES=4,5,6,7
python turngpt/train_conditional.py   --gpus -1 \
                        --batch_size 2 \
                        --accumulate_grad_batches 10\
                        --trp_projection_steps 1\
                        --max_length 100\
                        --strategy=ddp\
                        --tokenizer_punctuation_norm