
python turngpt/train_conditional.py   --gpus -1 \
                        --batch_size 3 \
                        --accumulate_grad_batches 10\
                        --trp_projection_steps 1\
			--max_length 100\
			--tokenizer_punctuation_norm\
			--dataset daily_dialog
		#	--fast_dev_run 10\
                       # --strategy=ddp

# export CUDA_VISIBLE_DEVICES=7
# python turngpt/model.py

# export CUDA_VISIBLE_DEVICES=5
# python turngpt/train.py   --gpus 1 \
#                         --batch_size 4 \
#                         --accumulate_grad_batches 10\
#                         --trp_projection_steps 1
