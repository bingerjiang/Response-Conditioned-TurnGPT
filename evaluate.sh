#export PYTHONPATH="/home/binger/miniconda3/envs/turngpt/bin/python:/home/binger/repos/futureTurnGPT"

# export CUDA_VISIBLE_DEVICES=3
# python futureturngpt/eval.py    --gpus 1\
#                                 --trp_projection_steps 1\
#                                 --batch_size 4\
#                                 --accumulate_grad_batches 10\
#                                 --max_length 200\
#                                 --num_workers 0\
#                                 --num_proc 4\
#                                 --datasets 'curiosity_dialogs'\
#                                 --trp_eval_type 'trp_proj'\
#                                 --calculate_metric

                                

# export CUDA_VISIBLE_DEVICES=1
# python futureturngpt/eval.py    --gpus 1\
#                                 --trp_projection_steps 1\
#                                 --batch_size 4\
#                                 --accumulate_grad_batches 10\
#                                 --max_length 120\
#                                 --num_proc 4\
#                                 --datasets 'daily_dialog'

# 11/14 threshold. but code is wrong
# export CUDA_VISIBLE_DEVICES=1
# python futureturngpt/eval.py    --gpus 1\
#                                 --trp_projection_steps 1\
#                                 --batch_size 2\
#                                 --accumulate_grad_batches 10\
#                                 --max_length 200\
#                                 --num_workers 0\
#                                 --num_proc 4\
#                                 --trp_eval_type 'trp_proj'\
#                                 --calculate_metric
                                #--datasets 'taskmaster1' 'taskmaster2' 'taskmaster3'\
# calculate threshold
# export CUDA_VISIBLE_DEVICES=2
# python futureturngpt/eval.py    --gpus 1\
#                                 --trp_projection_steps 1\
#                                 --batch_size 2\
#                                 --accumulate_grad_batches 10\
#                                 --max_length 180\
#                                 --num_workers 0\
#                                 --num_proc 4\
#                                 --trp_eval_type 'trp_proj'\
#                                 --calculate_metric                                

# export CUDA_VISIBLE_DEVICES=3
# python futureturngpt/eval.py    --gpus 1\
#                                 --trp_projection_steps 1\
#                                 --batch_size 2\
#                                 --accumulate_grad_batches 10\
#                                 --max_length 180\
#                                 --num_workers 0\
#                                 --num_proc 4\
#                                 --trp_eval_type 'trp_proj'\
                               # --datasets 'daily_dialog'\
                                #--calculate_ppl                               
#" you mean i have to wear the same thing every day mom<ts>hi brittany what are you doing with all of your clothes on your bed<ts> i'm trying to decide what to wear to school the first day<ts> oh a mom didn't tell you<ts> didn't tell me what what<ts> this bs'school you're going to is going to make your life easy<ts> what are you talking about brother spill it<ts> uniforms sis no more worrying about appearances<ts>"
#" you're welcome<ts>may i recommend you tsingtao beer<ts> tsingtao beer<ts> yes sir it's one of the best beers in china<ts> really<ts> yes the beer is brewed by using carefully selected malts rice hops and natural water from the lao mountain<ts> how about its taste<ts> fine sir<ts> that sounds great two tsingtao beers please<ts> tin or bottle<ts> tin please<ts> would you like it on the rocks sir<ts> no thank you<ts>"


export CUDA_VISIBLE_DEVICES=0
python futureturngpt/eval.py    --gpus 1\
                                --trp_projection_steps 1\
                                --batch_size 2\
                                --accumulate_grad_batches 10\
                                --max_length 180\
                                --num_workers 0\
                                --num_proc 4\
                                --trp_eval_type 'trp_proj'\
                                --collate_fn_type 'future_sod'\
                                --overwrite True\
                                --calculate_metric\
                                #--datasets 'meta_woz' 'multi_woz_v22' 'taskmaster1' 'taskmaster2' 'taskmaster3' 'daily_dialog' 'curiosity_dialogs' \
                                #--calculate_ppl\
                                