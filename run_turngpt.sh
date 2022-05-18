export CUDA_VISIBLE_DEVICES=4,5,6,7
python turngpt/train.py   --gpus 4 \
                        --batch_size 3 \
                        --accumulate_grad_batches 10\
                        --trp_projection_steps 1\
                        --strategy=ddp\
                        #--fast_dev_run 10