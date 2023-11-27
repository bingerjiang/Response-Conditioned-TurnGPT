from argparse import ArgumentParser
from os import makedirs
from os.path import join
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import pytorch_lightning as pl

from datasets_turntaking import ConversationalDM
from datasets_turntaking.conversational.utils import load_multiple_datasets
from datasets_turntaking import *
import futureturngpt
from futureturngpt.model import TurnGPT, TurnGPTWandbCallbacks
#from turngpt.conditional_model import *
import numpy as np
import pdb
from os import environ
import torch
local_rank = environ.get("LOCAL_RANK", 0)
#PROJECT = "TurnGPT"
#SAVE_DIR = "runs/TurnGPT"

def dm_prepare_conditional():
    '''
    b2: actually, probably don't need this
    '''
    print('test overriding method')
    
    for split in ["train", "validation", "test"]:
        split_path = self.get_split_path(split)

        if (
            self.overwrite
            or not self.load_from_cache_file
            or not exists(split_path)
            or len(listdir(split_path)) == 0
        ):

            # Remove if it exists in order to overwrite
            if self.overwrite and exists(split_path):
                shutil.rmtree(split_path)

            dsets = load_multiple_datasets(self.datasets, split)
            dataset = concatenate_datasets(dsets)
            print("filter empty turns")
            dataset = dataset.filter(self.filter_empty_turns)
            dataset = dataset.map(
                self.encode,
                batched=True,
                load_from_cache_file=self.load_from_cache_file,
                num_proc=self.num_proc,
            )
            dataset.set_format(type="torch")
            dataset.save_to_disk(split_path)

def default_logger_callbacks(name, args, callbacks):
    makedirs(args.SAVE_DIR, exist_ok=True)
    logger = WandbLogger(
        save_dir=args.SAVE_DIR,
        project=args.PROJECT,
        name=name + args.name_info,
        log_model=True,
    )

    callbacks.append(
        ModelCheckpoint(
            mode="min",
            monitor="val_loss",
        )
    )
    
    if not args.trial_run:
        print(f"Early stopping (patience={args.patience})")
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            strict=True,  # crash if "monitor" is not found in val metrics
            verbose=True,
        )
        callbacks.append(early_stop_callback)
    return logger, callbacks

#def take_turn (model, out, sublist_ts_idx):
        

def eval():
    #torch.multiprocessing.set_start_method('spawn')


    parser = ArgumentParser()
    parser = TurnGPT.add_model_specific_args(parser)
    parser = ConversationalDM.add_data_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--name_info", type=str, default="")
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument('--PROJECT', type=str, default='conditional_turngpt')
    parser.add_argument('--SAVE_DIR', type=str, default='runs/conditional_turngpt')
    parser.add_argument('--trial_run', action="store_true")
    parser.add_argument('--evaluate', action="store_true")
    #parser.add_argument("--calculate_metric", action='store_true', help='calculate acc, only use when run validation alone')

    args = parser.parse_args()

    pl.seed_everything(args.seed)
    print('--- args ---')
    print(args)
    #pdb.set_trace()
    # Model
    print("Loading Model...")
    torch.set_grad_enabled(False)
    # chpt = join(
    #     'runs/futureTurnGPT/futureTurnGPT_33671d2o/epoch=2_val_loss=1.1248.ckpt'
    # )
    chpt = join (
        'runs/futureTurnGPT/epoch=9_val_loss=1.7143.ckpt'
    )
    #pdb.set_trace()
    #model = TurnGPT.load_from_checkpoint(chpt).to("cuda")
    model = TurnGPT.load_from_checkpoint(chpt)
    model.trp_eval_type = args.trp_eval_type
    model.calculate_metric = True
    #print(args)
    model.print_parameters()
    model.eval()
    # DataModule
    use_dm = False
    if use_dm:
        dm = ConversationalDM(
            datasets=args.datasets,
            tokenizer=model.tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            savepath=args.savepath,
            overwrite=args.overwrite,
            load_from_cache_file=args.load_from_cache_file,
            num_proc=args.num_proc,
        )
        #pdb.set_trace()
        #dm.prepare_data = dm_prepare_conditional
        dm.prepare_data()
    #dm.setup()
    #pdb.set_trace()
    # Callbacks & Logger
    logger = None
    callbacks = None

    # this should be handled automatically with pytorch_lightning?
    
    callbacks = [TurnGPTWandbCallbacks()]
    logger, callbacks = default_logger_callbacks(
        name=model.run_name, args=args, callbacks=callbacks
    )

    # Trainer
    if args.trial_run:
        trainer = pl.Trainer.from_argparse_args(
            args=args,
            logger=logger,
            callbacks=callbacks,
            overfit_batches=0.001
        )
    else:
        trainer = pl.Trainer.from_argparse_args(
            args=args,
            logger=logger,
            callbacks=callbacks,
        #overfit_batches=0.001
        )
   # pdb.set_trace()

    # use the best lr suggested
    #lr_finder = trainer.tuner.lr_find(model,datamodule=dm)
    #new_lr = lr_finder.suggestion()
    
    
    ## re-initialize Trainer, because the following doesn't work:
    # model.hparams.lr = new_lr

    #trainer.validate(model, datamodule=dm)
    interval = 0.01
    thresholds = np.arange(0, 1, interval)
    #thresholds = [0.38]
    threshold_results_init = dict()
    for threshold in thresholds:
        threshold_results_init[threshold] = 0
    
    model.threshold_results_total = threshold_results_init
    #pdb.set_trace()
    model.n_turns = 0 # initialize number of turns
    model.turnlevel_max = 0
    model.calculate_metric = args.calculate_metric
    model.trp_eval_type = args.trp_eval_type
    
    #trainer.validate(model, datamodule=dm)
    
    turn_list = [
        #['What did you do yesterday?', 'I went hiking with my friends John and Mary. What did you do yesterday?', 'I was painting the wall of my garage.'],
        #['What did you do yesterday?', 'I went hiking with my friends John and Mary.', 'That sounds cool!'],
        #['What did you do yesterday?', 'I went hiking with my friends John and Mary. What did you do yesterday?', 'That sounds cool!'],
        #['What did you do yesterday?', 'I went hiking with my friends John and Mary. What did you do yesterday?', 'Are you okay?'],
        #['What did you order?', 'What we always have for brunch here, tuna sandwiches, fries, pudding, and coffee.', 'That sounds good!'],
        #['What did you order?', 'What we always have for brunch here, tuna sandwiches, fries, pudding, and coffee.', 'Are you okay?'],
        #['What did you order?', 'What we always have for brunch here, do you want anything else today?', 'No, that sounds good!']
        ['Yesterday we met in the park.','Okay, when will you meet again?','tomorrow.']
    ]
    import torch.nn as nn
    loss_fn = nn.BCEWithLogitsLoss()
    
    num_below_threshold = 0
    future_utt_ppl = []
    calculate_ppl = True
    turngpt_path = "/home/binger/repos/TurnGPT/runs/TurnGPT/TurnGPT_2cr6pudn/epoch=10_val_loss=1.7908.ckpt"
    if calculate_ppl:
        with torch.no_grad():
            from original_turngpt.model import originalTurnGPT, originalTurnGPTWandbCallbacks
            threshold = 0.42 # future turngpt
            #threshold_2 = 0.38 # turngpt
            original_turngpt = originalTurnGPT.load_from_checkpoint(
                join(turngpt_path)
            )
           
            original_turngpt.eval()
            model.transformer.config.output_attentions = True
            out = model.string_list_to_trp_inputid(turn_list)
            ts_idx = [(sublist == model.tokenizer.eos_token_id).nonzero(as_tuple=True)[0] for sublist in out['input_ids']]
            for idx, sublist_ts_idx in enumerate(ts_idx):
                current_utt_idx_start = sublist_ts_idx[-2]
                current_utt_prob = out['trp_probs'][idx][current_utt_idx_start+1:]
                current_utt_ids = out['input_ids'][idx][current_utt_idx_start+1:]

                pred_turnshift_idx = next(x[0] for x in enumerate(current_utt_prob) if x[1] > threshold)
                if pred_turnshift_idx is None:
                    # doesn't predict ts for this utterance, skip
                    num_below_threshold +=1
                    continue
                    
                else:    
                    #pred_utt_ids = current_utt_ids[:pred_turnshift_idx+1]
                    #pred_utt_ids.append(model.tokenizer.eos_token_id)
                    future_utt_ids = out['input_ids'][idx][:sublist_ts_idx[0]+1]
                    context_curr_utt_ids = out['input_ids'][idx][sublist_ts_idx[0]+1:sublist_ts_idx[-2]+pred_turnshift_idx+1]
                    if context_curr_utt_ids[-1] != model.tokenizer.eos_token_id:
                        #just insert <ts> token if there isn't one
                        context_curr_utt_ids = torch.cat(
                            (context_curr_utt_ids, torch.LongTensor([model.tokenizer.eos_token_id])), 0
                        )
                        #context_curr_utt_ids.append(model.tokenizer.eos_token_id)
                    new_dialog_ids = torch.cat((context_curr_utt_ids, future_utt_ids), 0) # add future utt back
                    new_dialog = model.tokenizer.decode(new_dialog_ids)
                    
                    ## use TURNGPT to evaluate
                    turngpt_out = original_turngpt.string_list_to_trp_inputid(new_dialog)
                    ts_idx_turngpt = [(sublist == model.tokenizer.eos_token_id).nonzero(as_tuple=True)[0] for sublist in turngpt_out['input_ids']]
                    
                    future_utt_trp_prob = turngpt_out['trp_probs'][:,ts_idx_turngpt[0][-2]+1:].squeeze(0)
                    # doesn't care about final <ts>
                    future_utt_trp_prob = future_utt_trp_prob[:-1]
                    future_labels = torch.zeros(len(future_utt_trp_prob))
                    future_labels[-1] =1
                    loss = loss_fn(future_utt_trp_prob, future_labels.float())
                    ppl = torch.exp(loss)
                    future_utt_ppl.append(ppl)
                    #pdb.set_trace()
    pdb.set_trace()


if __name__ == "__main__":
    eval()
