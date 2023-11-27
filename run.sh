export PYTHONPATH="/home/binger/miniconda3/envs/turngpt/bin/python:/home/binger/repos/futureTurnGPT"

# export CUDA_VISIBLE_DEVICES=7
# python turngpt/train.py   --gpus 1 \
#                         --batch_size 2 \
#                         --accumulate_grad_batches 10

# export CUDA_VISIBLE_DEVICES=7
# python turngpt/model.py

# export CUDA_VISIBLE_DEVICES=1
# python futureturngpt/train.py   --gpus 1 \
#                         --batch_size 2 \
#                         --accumulate_grad_batches 10\
#                         --trp_projection_steps 1\
#                         --max_length 50\
#                         --num_proc 4\
#                         --num_workers 0\
#                         --overwrite True\
#                         --datasets 'daily_dialog'
                        #--fast_dev_run 10

# older kind of collate_fn, works
# export CUDA_VISIBLE_DEVICES=2
# python futureturngpt/train.py   --gpus 1 \
#                         --batch_size 2 \
#                         --accumulate_grad_batches 10\
#                         --trp_projection_steps 1\
#                         --max_length 50\
#                         --num_proc 4\
#                         --num_workers 0\
#                         --overwrite True\
#                         --datasets 'daily_dialog'

# try new again
# export CUDA_VISIBLE_DEVICES=1
# python futureturngpt/train.py   --gpus 1 \
#                         --batch_size 2 \
#                         --accumulate_grad_batches 10\
#                         --trp_projection_steps 1\
#                         --max_length 100\
#                         --num_proc 4\
#                         --num_workers 0\
#                         --overwrite True\
#                         --datasets 'daily_dialog'

# export CUDA_VISIBLE_DEVICES=1
# python futureturngpt/train.py   --gpus 1 \
#                         --batch_size 2 \
#                         --accumulate_grad_batches 10\
#                         --trp_projection_steps 1\
#                         --max_length 100\
#                         --num_proc 4\
#                         --num_workers 0\
#                         --overwrite True\
                        #--datasets 'daily_dialog'

# test mask_current_token
# export CUDA_VISIBLE_DEVICES=3
# python futureturngpt/train.py   --gpus 1 \
#                         --batch_size 2 \
#                         --accumulate_grad_batches 10\
#                         --trp_projection_steps 1\
#                         --max_length 100\
#                         --num_proc 4\
#                         --num_workers 0\
#                         --overwrite True\
#                         --datasets 'daily_dialog'

## 1217 test sod
# export CUDA_VISIBLE_DEVICES=0
# python futureturngpt/train.py   --gpus 1 \
#                         --batch_size 2 \
#                         --accumulate_grad_batches 10\
#                         --trp_projection_steps 1\
#                         --max_length 100\
#                         --num_proc 4\
#                         --num_workers 0\
#                         --overwrite True\
# 			--collate_fn_type 'future_sod'\
# 			--datasets 'curiosity_dialogs'
                       
## 1217 
# export CUDA_VISIBLE_DEVICES=0
# python futureturngpt/train.py   --gpus 1 \
#                         --batch_size 2 \
#                         --accumulate_grad_batches 10\
#                         --trp_projection_steps 1\
#                         --max_length 180\
#                         --num_proc 4\
#                         --num_workers 0\
#                         --overwrite True\
# 			--collate_fn_type 'future_sod'\

#1218 continue
# export CUDA_VISIBLE_DEVICES=0
# python futureturngpt/train.py   --gpus 1 \
#                         --batch_size 2 \
#                         --accumulate_grad_batches 10\
#                         --trp_projection_steps 1\
#                         --max_length 180\
#                         --num_proc 4\
#                         --num_workers 0\
#                         --overwrite True\
# 			--collate_fn_type 'future_sod'\
#                         --continue_train_path '/home/binger/repo/futureTurnGPT/runs/futureTurnGPT/futureTurnGPT_3sdnm1mo/epoch=4_val_loss=0.2636.ckpt'

# export CUDA_VISIBLE_DEVICES=0
# python futureturngpt/train.py   --gpus 1 \
#                         --batch_size 2 \
#                         --accumulate_grad_batches 10\
#                         --trp_projection_steps 1\
#                         --max_length 180\
#                         --num_proc 4\
#                         --num_workers 0\
#                         --overwrite True\
# 			--collate_fn_type 'future'\


# 0115 drop incomplete future utt
# export CUDA_VISIBLE_DEVICES=0
# python futureturngpt/train.py   --gpus 1 \
#                         --batch_size 2 \
#                         --accumulate_grad_batches 10\
#                         --trp_projection_steps 1\
#                         --max_length 180\
#                         --num_proc 4\
#                         --num_workers 0\
#                         --overwrite True\
# 			--collate_fn_type 'future'\
                        #--continue_train_path '/home/binger/repo/futureTurnGPT/runs/futureTurnGPT/futureTurnGPT_3qrpr59m_epoch=3_val_loss=0.2666.ckpt'
# 0116 train from start
# export CUDA_VISIBLE_DEVICES=0
# python futureturngpt/train.py   --gpus 1 \
#                         --batch_size 2 \
#                         --accumulate_grad_batches 10\
#                         --trp_projection_steps 1\
#                         --max_length 180\
#                         --num_proc 4\
#                         --num_workers 0\
#                         --overwrite True\
# 			--collate_fn_type 'future'\

export CUDA_VISIBLE_DEVICES=0
python futureturngpt/train.py   --gpus 1 \
                        --batch_size 2 \
                        --accumulate_grad_batches 10\
                        --trp_projection_steps 1\
                        --max_length 180\
                        --num_proc 4\
                        --num_workers 0\
                        --overwrite True\
			--collate_fn_type 'future'\
                        --datasets 'taskmaster1' 'taskmaster2' 'taskmaster3' 'meta_woz' 'multi_woz_v22'
