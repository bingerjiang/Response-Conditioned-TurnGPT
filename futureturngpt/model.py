from argparse import ArgumentParser
import matplotlib as mpl
import matplotlib.pyplot as plt
import wandb
from os.path import join

from transformers import GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModelOutput
import pytorch_lightning as pl
import torch
import torch.nn as nn

from original_turngpt.model import originalTurnGPT, originalTurnGPTWandbCallbacks
from futureturngpt.generation import generate
from futureturngpt.plot_utils import plot_trp
from futureturngpt.projection_labeler import ProjectionLabeler
from futureturngpt.tokenizer import SpokenDialogTokenizer

import pdb
import numpy as np
mpl.use("agg")
from torchmetrics.classification import Accuracy

import openai
openai.api_key = "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"


def load_transformer(
    pretrained_model_name_or_path="gpt2", pretrained=True, **model_kwargs
):
    """Load transformer model. If `pretrained` then initialize with pretrained weights otherwise start from scratch"""

    implemented = [
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "microsoft/DialoGPT-small",
        "microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-large",
    ]

    update_on_pretrain = ["embd_pdrop", "attn_pdrop", "resid_pdrop"]

    if not (
        "gpt2" in pretrained_model_name_or_path.lower()
        or "dialogpt" in pretrained_model_name_or_path.lower()
    ):
        raise NotImplementedError(
            f"pretrained_model_name_or_path=`{pretrained_model_name_or_path}` is Not implemented! Please use: {implemented}"
        )

    config = GPT2Config.from_pretrained(pretrained_model_name_or_path)
    
    if pretrained:
        # Update only certain parameters
        for k, v in model_kwargs.items():
            if k in update_on_pretrain and v is not None:
                config.update({k: v})
        transformer = GPT2LMHeadModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, config=config
        )
    else:
        for k, v in model_kwargs.items():
            if v is not None:
                config.update({k: v})
        transformer = GPT2LMHeadModel(config=config)
    return transformer


class Utils:
    tokenizer: SpokenDialogTokenizer

    def idx_to_string(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        st = self.tokenizer.convert_ids_to_tokens(idx)
        #pdb.set_trace()
        st = self.tokenizer.convert_tokens_to_string(
            st.strip()
        )  # remove prefix space/symbol
        
        return st

    def get_trp(self, x):
        return x[..., self.tokenizer.eos_token_id]
    

    def tokenize_strings(self, string_or_list, add_post_eos_token=False):
        if isinstance(string_or_list, str) and add_post_eos_token:
            if not string_or_list.strip().endswith(self.tokenizer.eos_token):
                string_or_list += self.tokenizer.eos_token

        t = self.tokenizer(string_or_list, return_tensors="pt")
        if not isinstance(t["input_ids"], torch.Tensor):
            input_ids, speaker_ids = [], []
            
            input_ids = t["input_ids"]
            speaker_ids =  t["speaker_ids"]
            input_ids_arranged = list()
            speaker_ids_arranged = list()
            
            
            for idx, dialog in enumerate(input_ids):
                if dialog[-1] != self.tokenizer.eos_token_id: 
                    
                    dialog.append(self.tokenizer.eos_token_id)
                
                n_subdialogs = dialog.count(self.tokenizer.eos_token_id)
                ts_indices = [i for i, x in enumerate(dialog) if x == self.tokenizer.eos_token_id]
                
                i = 0
                
                next_utt_input_ids = dialog[ts_indices[-2]+1:]
                #next_utt_input_ids.insert(0, self.tokenizer.bos_token_id)
                past_current_utt_input_ids = dialog[:ts_indices[-2]+1]
                new_utt = next_utt_input_ids+ past_current_utt_input_ids
                input_ids_arranged.append(torch.tensor(new_utt))

                next_utt_speaker_ids = [self.tokenizer.future_token_id] * len(next_utt_input_ids)
                past_current_utt_speaker_ids = speaker_ids[idx][:ts_indices[-2]+1]
                speaker_ids_arranged.append(torch.tensor(next_utt_speaker_ids+ past_current_utt_speaker_ids))
                
                dialog = dialog[:ts_indices[-2]+1]
                
                    
            
            tmp_inp = []
            tmp_sp = []
            for inp, sp in zip(input_ids_arranged, speaker_ids_arranged):
               tmp_inp.append(torch.tensor(inp))
               tmp_sp.append(torch.tensor(sp))
            tmp = self.tokenizer.pad({"input_ids": tmp_inp})

            t["input_ids"] = tmp["input_ids"]
            t["attention_mask"] = tmp["attention_mask"]
            t["speaker_ids"] = self.tokenizer.pad({"input_ids": tmp_sp})["input_ids"]
        for k, v in t.items():
            t[k] = v.to(self.device)
        return t


    def tokenize_strings_bos(self, string_or_list, add_post_eos_token=False):
        if isinstance(string_or_list, str) and add_post_eos_token:
            if not string_or_list.strip().endswith(self.tokenizer.eos_token):
                string_or_list += self.tokenizer.eos_token

        t = self.tokenizer(string_or_list, return_tensors="pt")
        if not isinstance(t["input_ids"], torch.Tensor):
            input_ids, speaker_ids = [], []
            
            input_ids = t["input_ids"]
            speaker_ids =  t["speaker_ids"]
            input_ids_arranged = list()
            speaker_ids_arranged = list()
            
            
            for idx, dialog in enumerate(input_ids):
                if dialog[-1] != self.tokenizer.eos_token_id: 
                    
                    dialog.append(self.tokenizer.eos_token_id)
                
                n_subdialogs = dialog.count(self.tokenizer.eos_token_id)
                ts_indices = [i for i, x in enumerate(dialog) if x == self.tokenizer.eos_token_id]
                
                i = 0
                
                next_utt_input_ids = dialog[ts_indices[-2]+1:]
                next_utt_input_ids.insert(0, self.tokenizer.bos_token_id)
                past_current_utt_input_ids = dialog[:ts_indices[-2]+1]
                new_utt = next_utt_input_ids+ past_current_utt_input_ids
                input_ids_arranged.append(torch.tensor(new_utt))

                next_utt_speaker_ids = [self.tokenizer.future_token_id] * len(next_utt_input_ids)
                past_current_utt_speaker_ids = speaker_ids[idx][:ts_indices[-2]+1]
                speaker_ids_arranged.append(torch.tensor(next_utt_speaker_ids+ past_current_utt_speaker_ids))
                
                dialog = dialog[:ts_indices[-2]+1]
                
                    
            
            tmp_inp = []
            tmp_sp = []
            for inp, sp in zip(input_ids_arranged, speaker_ids_arranged):
               tmp_inp.append(torch.tensor(inp))
               tmp_sp.append(torch.tensor(sp))
            tmp = self.tokenizer.pad({"input_ids": tmp_inp})

            t["input_ids"] = tmp["input_ids"]
            t["attention_mask"] = tmp["attention_mask"]
            t["speaker_ids"] = self.tokenizer.pad({"input_ids": tmp_sp})["input_ids"]
        for k, v in t.items():
            t[k] = v.to(self.device)
        return t
    def tokenize_strings_old(self, string_or_list, add_post_eos_token=False):
        if isinstance(string_or_list, str) and add_post_eos_token:
            if not string_or_list.strip().endswith(self.tokenizer.eos_token):
                string_or_list += self.tokenizer.eos_token

        t = self.tokenizer(string_or_list, return_tensors="pt")
        if not isinstance(t["input_ids"], torch.Tensor):
            input_ids, speaker_ids = [], []
            #pdb.set_trace()
            
            # for inp, sp in zip(t["input_ids"], t["speaker_ids"]):
            #    input_ids.append(torch.tensor(inp))
            #    speaker_ids.append(torch.tensor(sp))
            # 可能是没有一个个torch.tensor的原因？
            #input_ids = tmp_inp
            #speaker_ids = tmp_sp
            input_ids = t["input_ids"]
            speaker_ids =  t["speaker_ids"]
            input_ids_arranged = list()
            speaker_ids_arranged = list()
            
            
            for idx, dialog in enumerate(input_ids):
                if dialog[-1] != self.tokenizer.eos_token_id: 
                    #dialog = dialog[:-1]
                #dialog_tmp = dialog.tolist()
                    dialog.append(self.tokenizer.eos_token_id)
                
                n_subdialogs = dialog.count(self.tokenizer.eos_token_id)
                ts_indices = [i for i, x in enumerate(dialog) if x == self.tokenizer.eos_token_id]
                
                i = 0
                while i < n_subdialogs-1:
                    next_utt_input_ids = dialog[ts_indices[-2]+1:]
                    past_current_utt_input_ids = dialog[:ts_indices[-2]+1]
                    #new_utt = torch.cat((next_utt_input_ids, past_current_utt_input_ids),0)
                    new_utt = next_utt_input_ids+ past_current_utt_input_ids
                    input_ids_arranged.append(torch.tensor(new_utt))
                    #pdb.set_trace()
                    next_utt_speaker_ids = [self.tokenizer.future_token_id] * len(next_utt_input_ids)
                    past_current_utt_speaker_ids = speaker_ids[idx][:ts_indices[-2]+1]
                    speaker_ids_arranged.append(torch.tensor(next_utt_speaker_ids+ past_current_utt_speaker_ids))
                    
                    dialog = dialog[:ts_indices[-2]+1]
                    ts_indices.pop()
                    #pdb.set_trace()
                    i += 1
            
            #pdb.set_trace()
            tmp_inp = []
            tmp_sp = []
            for inp, sp in zip(input_ids_arranged, speaker_ids_arranged):
               tmp_inp.append(torch.tensor(inp))
               tmp_sp.append(torch.tensor(sp))
            tmp = self.tokenizer.pad({"input_ids": tmp_inp})
            # t["input_ids"] = self.tokenizer.pad(
            #     {"input_ids": tmp_inp}
            # )
            # t["speaker_ids"] = self.tokenizer.pad(
            #     {"input_ids": tmp_sp}
            # )["input_ids"]
            t["input_ids"] = tmp["input_ids"]
            t["attention_mask"] = tmp["attention_mask"]
            t["speaker_ids"] = self.tokenizer.pad({"input_ids": tmp_sp})["input_ids"]
        #pdb.set_trace()
        for k, v in t.items():
           # pdb.set_trace()
            t[k] = v.to(self.device)
        return t

    def get_tokens(self, input_ids):
        def inner(input_ids):
            inner_tokens = []
            for idx in input_ids:
                inner_tokens.append(self.idx_to_string(idx))
            return inner_tokens

        def outer(input_ids):
            tokens = []
            for batch in input_ids:
                tokens.append(inner(batch))
            return tokens

        tokens = None
        if isinstance(input_ids, torch.Tensor):
            if input_ids.ndim > 2:
                raise LookupError(
                    f"self.get_tokens not implemented for Tensor shape: {tuple(input_ids.shape)} (>2)"
                )
            elif input_ids.ndim == 2:
                tokens = outer(input_ids)
            else:
                tokens = inner(input_ids)
                for idx in input_ids:
                    tokens.append(self.idx_to_string(idx))
        elif isinstance(input_ids, list):
            if isinstance(input_ids[0], list):
                tokens = outer(input_ids)
            else:
                tokens = inner(input_ids)
        return tokens

    @torch.no_grad()
    def string_list_to_trp(
        self, string_or_list, add_post_eos_token=False, use_label=True, **model_kwargs
    ):
        t = self.tokenize_strings(string_or_list, add_post_eos_token=add_post_eos_token)
        #pdb.set_trace()
        if use_label:
            lm_labels = self.get_labels(t["input_ids"], mask=t['attention_mask'])
        proj_labels = None
        if self.trp_projection_steps > 0:
            proj_labels = self.get_projection_labels(
                t["input_ids"], mask=t["attention_mask"]
            )
        # Model
        if not use_label:
            out = self(t["input_ids"], speaker_ids=t["speaker_ids"], **model_kwargs)
        else:
            out = self(t["input_ids"], 
                       speaker_ids=t["speaker_ids"], 
                       labels=lm_labels,
                       mc_labels=proj_labels,
                       **model_kwargs)
        out["probs"] = out["logits"].softmax(dim=-1)
        out["trp_probs"] = self.get_trp(out["probs"])
        #pdb.set_trace()
        out["tokens"] = self.get_tokens(t["input_ids"])
        if "mc_logits" in out:
            out["trp_proj"] = out["mc_logits"].sigmoid()
        

        #pdb.set_trace()
        return out
    @torch.no_grad()
    def string_list_to_trp_inputid(
        self, string_or_list, add_post_eos_token=False, use_label=True,**model_kwargs
    ):
        t = self.tokenize_strings(string_or_list, add_post_eos_token=add_post_eos_token)
        if use_label:
            lm_labels = self.get_labels(t["input_ids"], mask=t['attention_mask'])
        proj_labels = None
        if self.trp_projection_steps > 0:
            proj_labels = self.get_projection_labels(
                t["input_ids"], mask=t["attention_mask"]
            )
        # Model
        if not use_label:
            out = self(t["input_ids"], speaker_ids=t["speaker_ids"], **model_kwargs)
        else:
            out = self(t["input_ids"], 
                       speaker_ids=t["speaker_ids"], 
                       labels=lm_labels,
                       mc_labels=proj_labels,
                       **model_kwargs)        
        out["probs"] = out["logits"].softmax(dim=-1)
        out["trp_probs"] = self.get_trp(out["probs"])
        out["tokens"] = self.get_tokens(t["input_ids"])
        out['input_ids'] = t['input_ids']
        if "mc_logits" in out:
            out["trp_proj"] = out["mc_logits"].sigmoid()
        #pdb.set_trace()
        return out    

class TurnGPTWandbCallbacks(pl.Callback):
    # turn_list = [
    #     ["yesterday we met in the park", "okay when will you meet again", "tomorrow"],
    #     [
    #         "Hello there I basically had the worst day of my life",
    #         "Oh no, what happened?",
    #         "Do you want the long or the short story?",
    #     ],
    # ]
    turn_list = [
        ['What did you do yesterday?', 'I went hiking with my friends John and Mary. What did you do?', 'I was painting the wall of my garage.'],
        ['What did you do yesterday?', 'I went hiking with my friends John and Mary.', 'That sounds cool!'],
        ['What did you do yesterday?', 'I went hiking with my friends John and Mary. What did you do?', 'That sounds cool!'],
        ['What did you do yesterday?', 'I went hiking with my friends John and Mary. What did you do?', 'Are you okay?'],
        ['What did you order?', 'What we always have for brunch here, tuna sandwiches, fries, pudding, and coffee.', 'That sounds good!'],
        ['What did you order?', 'What we always have for brunch here, tuna sandwiches, fries, pudding, and coffee.', 'Are you okay?'],
        ['What did you order?', 'What we always have for brunch here, do you want anything else today?', 'No, that sounds good!']
    ]
    def __init__(
        self,
        text_list=None,
        n_steps=200,
        n_generate=20,
        eos_token="<ts>",
        unk_token="<|endoftext|>",
    ):
        super().__init__()
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.text_list = text_list
        self.n_steps = n_steps
        self.n_generate = n_generate
        if self.text_list is None:
            self.text_list = self.turn_list

    def trp_plots(self, trainer, pl_module, name="TRP/example"):
        out = pl_module.string_list_to_trp(self.text_list)

        for b in range(out["trp_probs"].shape[0]):
            proj = out["trp_proj"][b].cpu() if "trp_proj" in out else None
            fig, _ = plot_trp(
                trp=out["trp_probs"][b].cpu(),
                proj=proj,
                text=out["tokens"][b],
                unk_token=pl_module.tokenizer.unk_token,
                eos_token=pl_module.tokenizer.eos_token,
                plot=False,
            )

            pl_module.logger.experiment.log(
                data={
                    f"{name}_{b}": wandb.Image(fig),
                    "global_step": trainer.global_step,
                },
            )
            plt.close("all")

    def generate(self, trainer, pl_module, name):
        gen = generate(
            pl_module,
            context=self.turn_list[-1],
            n_steps=self.n_steps,
            top_p=0.9,
            top_k=-1,
            n_trajectories=self.n_generate,
            strategy="sample",
            stop_at_eos=True,
        )
        # remove duplicates
        l = (gen["input_ids"][0] != -1).sum()
        G = {"tokens": [gen["tokens"][0]], "probs": [gen["probs"][0][:l].cpu()]}
        for i, g in enumerate(gen["tokens"][1:]):
            if g not in G["tokens"]:
                l = (gen["input_ids"][i] != -1).sum()
                G["tokens"].append(g)
                G["probs"].append(gen["probs"][i][:l].cpu())

        table = wandb.Table(
            columns=["context", "sample", "probs"],
            data=[
                ["... " + self.turn_list[-1][-1], toks, probs.tolist()]
                for toks, probs in zip(G["tokens"], G["probs"])
            ],
        )
        pl_module.logger.experiment.log(
            data={
                f"{name}": table,
                "global_step": trainer.global_step,
            },
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        #self.trp_plots(trainer, pl_module, name="TRP/example")
        #self.generate(trainer, pl_module, name="Gen")
        #pdb.set_trace()
        print(pl_module.calculate_metric)
        #pdb.set_trace()
        if pl_module.calculate_metric:
            try:
                best_key = max(pl_module.threshold_results_total, key=pl_module.threshold_results_total.get)
            except ValueError:
                pdb.set_trace()
            print('best threshold = ', best_key)
            print('n_correct',pl_module.threshold_results_total[best_key])
            print('total number of turns = ', pl_module.n_turns)
            #print('accuracy = ', pl_module.threshold_results_total/pl_module.n_turns)
            print('turnlevel max at end: ', pl_module.turnlevel_max)
        #pdb.set_trace()
        #threshold_results = pl_module.find_best_word_threshold(pl_module.y_hat, pl_module.y, 0.01)
        #self.attention_plot(trainer,pl_module, name='TRP/attention_end')
    def on_validation_epoch_start(self, trainer, pl_module):
        #pl_module.attention_plot(trainer,pl_module, name='TRP/attention_end')
        try:
            print(pl_module.threshold_results_total)
        except ValueError:
            pdb.set_trace()
    ## commented out b2 1217
    #def on_save_checkpoint(self, trainer, pl_module, *args, **kwargs):
        #self.trp_plots(trainer, pl_module, name="TRP-chpt/example")
        #self.generate(trainer, pl_module, name="Gen-chpt")


class TurnGPT(pl.LightningModule, Utils):
    """
    This is the code example of teaching and research.

    Add features to this model i.e. analysis of turn-taking.


    On training
    * call `model.initialize_special_embeddings()` to initialize <ts> = eos_token
    """

    def __init__(
        self,
        pretrained_model_name_or_path="gpt2",
        pretrained=True,
        trp_projection_steps=1,
        trp_projection_type="linear",
        omit_dialog_states=False,
        no_train_first_n=5,
        learning_rate=1e-4,
        weight_loss=False,
        weight_regular_token=0.5,
        weight_eos_token=1.0,
        n_turns = 0,
        trp_eval_type = None,
        threshold_results_total = None,
        turnlevel_max = None,
        calculate_metric = None,
        calculate_ppl = None,     
        original_turngpt = None,
        gpt3 = None,
        **model_kwargs,
    ):
        super().__init__()
        self.name_or_path = pretrained_model_name_or_path
        self.pretrained = pretrained

        # train parameters
        self.no_train_first_n = no_train_first_n
        self.learning_rate = learning_rate
        self.weight_loss = weight_loss
        self.weight_regular_token = weight_regular_token
        self.weight_eos_token = weight_eos_token
        self.omit_dialog_states = omit_dialog_states

        # Load `transformers` model
        self.transformer = load_transformer(
            pretrained_model_name_or_path, pretrained=pretrained, **model_kwargs
        )
        # load original turngpt model
        # self.original_turngpt = originalTurnGPT.load_from_checkpoint(
        #         join("/home/binger/repos/TurnGPT/runs/TurnGPT/TurnGPT_2cr6pudn/epoch=10_val_loss=1.7908.ckpt")
        #     )
        self.original_turngpt = original_turngpt
        self.gpt3 = gpt3
        # TRP projection head
        self.trp_projection_steps = trp_projection_steps
        if self.trp_projection_steps > 0:
            self.trp_projection_type = trp_projection_type
            hidden_size = self.transformer.config.hidden_size

            # MultiTask Head operating on n last hidden states
            if trp_projection_type.lower() == "attention":
                raise NotImplementedError()
            else:
                self.trp_projection_head = nn.Linear(hidden_size, 1)

        
        self.threshold_results_total = threshold_results_total
        self.turnlevel_max = turnlevel_max
        self.y = None
        self.y_hat = None
        self.trp_eval_type = trp_eval_type
        self.calculate_metric = calculate_metric
        self.calculate_ppl = calculate_ppl
        self.n_turns = n_turns
        self.threshold_results_total = threshold_results_total
        self.num_below_threshold = 0
        self.future_utt_ppl = 0
        self.num_future_utts = 0
        self.enable_attention = None
        ## following three: future utterance attentions
        self.furthest_idx_all = None
        self.rel_furthest_idx_all = None
        self.precent_future_attention = None
        #self.save_hyperparameters()
    @property
    def run_name(self):
        name = "futureTurnGPT"
        if self.trp_projection_steps > 0:
            name += f"_proj_{self.trp_projection_steps}"
        return name

    def init_tokenizer(self):
        # The tokenizer should always be a part of the model
        self.tokenizer = SpokenDialogTokenizer(self.name_or_path)

        # Add extra embeddings for custom tokens
        # Optional: Initialize <ts> to be close to punctuation tokens.
        self.transformer.resize_token_embeddings(new_num_tokens=len(self.tokenizer))

    def initialize_special_embeddings(self, tokens=["!", "?", "."]):
        """
        Initialize `eos_token` as the average of `tokens`.

        By default (or looking at <speaker1/2>) the embeddings are initalized to m=0, std=0.02
        """
        ts = self.tokenizer.eos_token_id
        # pre = self.transformer.transformer.wte.weight[ts].clone()
        with torch.no_grad():
            ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))
            avg_emb = self.transformer.transformer.wte(ids).mean(0)
            self.transformer.transformer.wte.weight.data[ts] = avg_emb
        # post = self.transformer.transformer.wte.weight[ts]
        # print(pre == post)
        print(f"Initalized {self.tokenizer.eos_token} -> avg({tokens})")

    def print_parameters(self):
        print("")
        print("futureTurnGPT")
        print("name_or_path: ", self.name_or_path)
        print("learning_rate: ", self.learning_rate)
        print("weight_loss: ", self.weight_loss)
        if self.weight_loss:
            print("weight_regular_token: ", self.weight_regular_token)
            print("weight_eos_token: ", self.weight_eos_token)
        if self.trp_projection_steps > 0:
            print("trp_projection_steps: ", self.trp_projection_steps)
            print("trp_projection_type: ", self.trp_projection_type)
        print()

    def get_labels(self, input_ids, mask, value=-100):
        """Don't shift the labels (happens internally)"""
        labels = input_ids.clone()
        labels[torch.logical_not(mask)] = value

        if self.no_train_first_n > 0:
            labels[:, : self.no_train_first_n] = value
        return labels

    def get_projection_labels(self, input_ids, mask, value=-100):
        labeler = ProjectionLabeler(
            projection_steps=self.trp_projection_steps,
            token_id=self.tokenizer.eos_token_id,
        ).to(self.device)
        proj_labels = labeler(input_ids)
        proj_labels[torch.logical_not(mask)] = value
        if self.no_train_first_n > 0:
            proj_labels[:, : self.no_train_first_n] = value
        return proj_labels

    @torch.no_grad()
    def get_loss_weight(self):
        weight = (
            torch.ones(len(self.tokenizer), dtype=torch.float)
            * self.weight_regular_token
        )
        weight[self.tokenizer.eos_token_id] = self.weight_eos_token
        return weight.to(self.device)

    def cross_entropy_loss(self, logits, labels, reduction="mean"):
        """
        Taken from GPT2LMHeadModel in:

          https://github.com/huggingface/transformers/blob/91ff480e2693f36b11aaebc4e9cc79e4e3c049da/src/transformers/models/gpt2/modeling_gpt2.py#L968

        How to weight CE-Loss?

            https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731/10

        Using the custom weights gets a total loss that is larger than without?
        I don't want the model to train as much on the new type of text it receives but to learn the `eos_token` better.
        I assume that the loss should be less if I scale down the normal tokens loss...

        `CrossEntropyLoss` seems to normalize the loss if not `reduction=None` which account for the above behaviour.

        Instead we use `reduction=none` and simply average over the weighted loss values.
        Given that `weight_regular_token` < 1 and `weight_eos_token` >=1 we get a lower loss when weighting.

        Is this mathematically sound? well that is the question.

        I guess one could simply scale the `weight_eos_token > 1` while `weight_regular_token=1` and use a smaller learning rate?
        """
        weight = None
        if self.weight_loss:
            weight = self.get_loss_weight()

        loss_fct = nn.CrossEntropyLoss(weight=weight, reduction="none")

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens and calc loss
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        if reduction != "none":
            loss = loss.mean()
        #pdb.set_trace()
        return loss
    def attention_plot (self, trainer, pl_module, name="TRP/attention", unk_token="<|endoftext|>"):
        out = pl_module.string_list_to_trp(self.text_list)
        attn = out['attentions']
        for b in range(attn.shape[0]):

            fig, ax = plt.subplots(
                attn.shape[1],  sharex=True, sharey=True, figsize=(12, 36)
            )
            text = out['tokens'][b]
            #text.insert(0, '<nextutt>')
            #text.pop()
            if text is not None:
                max_idx = len(text)
            for n, t in enumerate(text):
                if t == unk_token:
                    max_idx = n
                    break
            text = text[:max_idx]
            attn = attn[:max_idx]
            for n_head in range(attn.shape[1]):
                ax[n_head].imshow(
                    attn[b, n_head].cpu(),
                    aspect="auto",
                    origin="upper",
                    interpolation="none",
                    vmin=0,
                    vmax=1,
                    cmap="viridis",
                )
                ax[n_head].set_ylabel(f"Head {n_head}")
            x = torch.arange(len(text))
            ax[b].set_xticks(x)
            ax[b].set_yticks(x)
        
            plt.tight_layout()
            if text is not None:
                ax[b].set_xticklabels(text, rotation=60)
                ax[b].set_yticklabels(text, rotation=60)

                
            pl_module.logger.experiment.log(
                data={
                    f"{name}_{b}": wandb.Image(fig),
                    "global_step": trainer.global_step,
                },
            )
            #pdb.set_trace()
            plt.close("all")
    def bce_loss(self, logits, labels):
        """
        Simple BCELoss for binary trp projection

        Must extend this if multiple labels are to be used...
        """
        loss_fct = nn.BCEWithLogitsLoss()

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1]  # , :].contiguous()
        shift_labels = labels[..., 1:]  # .contiguous()

        # Manually select appropriate steps
        # Omit steps where label is -100 (like CrossEntropyLoss)
        indices_for_training = shift_labels != -100
        loss = loss_fct(
            torch.masked_select(shift_logits, indices_for_training),
            torch.masked_select(shift_labels, indices_for_training),
        )
        # shift_logits = torch.masked_select(shift_logits, indices_for_training)
        # shift_labels = torch.masked_select(shift_labels, indices_for_training)
        # loss = loss_fct(shift_logits, shift_labels)
        #pdb.set_trace()
        return loss
    def mask_future_token (self, labels, speaker_ids, value=-100):
        '''mask out future tokens based on speaker_ids'''
        future_token_idx = [(el == self.tokenizer.future_token_id).nonzero(as_tuple=True)[0] for el in speaker_ids]
        i = 0
        while i < len(future_token_idx):
            labels[i][future_token_idx[i]] = value
            i+=1
        #print('print masking future tokens')
        #pdb.set_trace()
        return labels
    def mask_current_token (self, input_ids, labels, value=-100):
        # needs fix
        ts_idx = [(sublist == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0] for sublist in input_ids]
        i = 0
        while i < len(ts_idx):
            
            if len(ts_idx[i])>1:
                if input_ids[i][-1] not in [50256, self.tokenizer.eos_token_id]:
                    print('truncated sent')
                    pdb.set_trace()
                pdb.set_trace()
                labels[i][ts_idx[i][-2]+1:] = value
            i += 1
        #print('print masking current token?')
        #pdb.set_trace()
        return labels
    def mask_context_token (self, input_ids, labels, value=-100):
        # needs fix
        ts_idx = [(sublist == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0] for sublist in input_ids]
        i = 0
        while i < len(ts_idx):
            
            if len(ts_idx[i])>1:
                if input_ids[i][-1] not in [50256, self.tokenizer.eos_token_id]:
                    print('truncated sent')
                    # ok, since it is current sentence it can't be truncated
                    labels[i][:ts_idx[i][-1]+1] = value
                else:
                    labels[i][:ts_idx[i][-2]+1] = value
            i += 1
        #print('print masking context token?')
        #pdb.set_trace()
        return labels
    def get_likelihood(self, logits, labels, pad_last=True, pad_first=False):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_probs = shift_logits.softmax(dim=-1)

        # likelihood = shift_probs[shift_labels]
        bn = torch.ones_like(shift_labels)
        bn[0] = 0

        seq = torch.arange(shift_labels.shape[-1])
        seq = torch.stack([seq] * shift_labels.shape[0])

        likelihood = shift_probs[bn.view(-1), seq.view(-1), shift_labels.view(-1)]
        likelihood = likelihood.view(shift_labels.shape)
        if pad_first:
            likelihood = torch.cat(
                [torch.zeros(likelihood.shape[0], 1), likelihood], dim=-1
            )
        elif pad_last:
            likelihood = torch.cat(
                [likelihood, torch.zeros(likelihood.shape[0], 1)], dim=-1
            )
        return likelihood
    def get_turnlevel_acc_max_at_end (self, out, input_ids = None):
        if input_ids is None:
            input_ids = out['input_ids']
        ts_idx = [(sublist == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0] for sublist in input_ids]
        ## process stuff (done in string_list_to_Trp)

        out["probs"] = out["logits"].softmax(dim=-1)
        out["trp_probs"] = self.get_trp(out["probs"])
        if "mc_logits" in out:
            out["trp_proj"] = out["mc_logits"].sigmoid()         
        correct_turns = 0

        #for b in range(len(ts_idx)):
        for b in range(out["trp_proj"].shape[0]):
            dialog_ts = ts_idx[b]
            ith_turn = 0
            while ith_turn < len(dialog_ts)-1:
                if ith_turn == 0:
                    start_idx = 0
                    end_idx = dialog_ts[ith_turn]
                else:
                    start_idx = dialog_ts[ith_turn-1]+1
                    end_idx = dialog_ts[ith_turn]
                trp_proj = out['trp_proj'][b].cpu()
                max_trp_idx = torch.max(trp_proj[start_idx:end_idx],0).indices
                if max_trp_idx == len(trp_proj[start_idx:end_idx])-1:
                    correct_turns +=1
                else:
                    sentence = self.tokenizer.decode(input_ids[b])

                ith_turn += 1
        return correct_turns
    def get_turnlevel_acc_max_at_end_future (self, out, input_ids = None):
        if input_ids is None:
            input_ids = out['input_ids']
        ts_idx = [(sublist == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0] for sublist in input_ids]
        ## process stuff (done in string_list_to_Trp)

        out["probs"] = out["logits"].softmax(dim=-1)
        out["trp_probs"] = self.get_trp(out["probs"])
        if "mc_logits" in out:
            out["trp_proj"] = out["mc_logits"].sigmoid()         
        correct_turns = 0

        #for b in range(len(ts_idx)):
        for b in range(out["trp_proj"].shape[0]):
            dialog_ts = ts_idx[b]
            start_idx = dialog_ts[-2]+1
            end_idx = dialog_ts[-1]
            trp_proj = out['trp_proj'][b].cpu()
            max_trp_idx = torch.max(trp_proj[start_idx:end_idx],0).indices
            if max_trp_idx == len(trp_proj[start_idx:end_idx])-1:
                    correct_turns +=1
            else:
                sentence = self.tokenizer.decode(input_ids[b])
            
            
        return correct_turns
    def get_trp_proj_word_with_tgt(self, logits, labels):
        """
        added by b2 sept 13
         returns the masked target
        retuns [y_hat, y] for calculating bacc (macro acc)
        """

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1]  # , :].contiguous()
        shift_labels = labels[..., 1:]  # .contiguous()

        # Manually select appropriate steps
        # Omit steps where label is -100 (like CrossEntropyLoss)
        indices_for_training = shift_labels != -100
        y_hat = torch.masked_select(shift_logits, indices_for_training).sigmoid()
        y = torch.masked_select(shift_labels, indices_for_training)

        #y_hat_new = [0 if el<threshold else 1 for el in y_hat]
       # pdb.set_trace()
        y_hat_new = torch.tensor(y_hat).to(y_hat.device)        
        
        return y_hat_new, y
    
    def find_best_threshold_turnlevel (self, out, input_ids=None):
        # interval = 0.01
        # thresholds = np.arange(0, 1, interval)
        # threshold_results = dict()
        # for threshold in thresholds:
        #     threshold_results[threshold] = 0
        #thresholds = [0.8, 0.7, 0.6, 0.5]
        #threshold_results = {0.5:0, 0.6:0, 0.7:0, 0.8:0}
        if input_ids is None:
            input_ids = out['input_ids']
        ts_idx = [(sublist == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0] for sublist in input_ids]

        
        ## process stuff (done in string_list_to_Trp)

        out["probs"] = out["logits"].softmax(dim=-1)
        out["trp_probs"] = self.get_trp(out["probs"])
        if "mc_logits" in out:
            out["trp_proj"] = out["mc_logits"].sigmoid()        
        
        
        if self.trp_eval_type not in ["trp_proj", "trp_probs"]:
            print("wrong trp_eval_type")
        else:
            out_trp = out[self.trp_eval_type]
                    
        n_turns = 0
        for b in range(out_trp.shape[0]):
            dialog_ts = ts_idx[b]
            n_turns += len(dialog_ts)-1
            for threshold in self.threshold_results_total.keys():
                ith_turn = 0              
                while ith_turn < len(dialog_ts)-1:
                    if ith_turn == 0:
                        start_idx = 0
                        end_idx = dialog_ts[ith_turn]
                    else:
                        start_idx = dialog_ts[ith_turn-1]+1
                        end_idx = dialog_ts[ith_turn]
                    trp_proj = out_trp[b].cpu()
                    current_utt = trp_proj[start_idx:end_idx]
                    idx_above_threshold = [i for i in range(len(current_utt)) if current_utt[i]>threshold]
                    #print(idx_above_threshold)
                    #pdb.set_trace()
                    if len(idx_above_threshold)==1 and idx_above_threshold[0] == len(current_utt)-1:
                    ## if only one satisfies and it is the last position where it is the ts
                    ## this is the only case it satisfies. add 1
                        self.threshold_results_total[threshold] += 1
                    #max_trp_idx = torch.max(trp_proj[start_idx:end_idx],0).indices
                    #max_trp = torch.max(trp_proj[start_idx:end_idx],0).values

                    ith_turn += 1
        self.n_turns += n_turns
        #return threshold_results, n_turns
    def find_best_threshold_turnlevel_future (self, out, input_ids=None):
        # interval = 0.01
        # thresholds = np.arange(0, 1, interval)
        # threshold_results = dict()
        # for threshold in thresholds:
        #     threshold_results[threshold] = 0
        #thresholds = [0.8, 0.7, 0.6, 0.5]
        #threshold_results = {0.5:0, 0.6:0, 0.7:0, 0.8:0}
        if input_ids is None:
            input_ids = out['input_ids']
        ts_idx = [(sublist == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0] for sublist in input_ids]

        
        ## process stuff (done in string_list_to_Trp)

        out["probs"] = out["logits"].softmax(dim=-1)
        out["trp_probs"] = self.get_trp(out["probs"])
        if "mc_logits" in out:
            out["trp_proj"] = out["mc_logits"].sigmoid()        
        
        
        if self.trp_eval_type not in ["trp_proj", "trp_probs"]:
            print("wrong trp_eval_type")
        else:
            out_trp = out[self.trp_eval_type]
        #pdb.set_trace()
        
        #n_turns = 0
        for b in range(out_trp.shape[0]):
            dialog_ts = ts_idx[b]
            
            for threshold in self.threshold_results_total.keys():
                # only one sentence per dialog
                
                #pdb.set_trace()
                # since it is current sentence it can't be truncated
                
                start_idx = dialog_ts[-2]+1
                
                end_idx = dialog_ts[-1]
                trp_proj = out_trp[b].cpu()
                current_utt = trp_proj[start_idx:end_idx] # doesnt include <ts>
                idx_above_threshold = [i for i in range(len(current_utt)) if current_utt[i]>threshold]
                if len(idx_above_threshold)==1 and idx_above_threshold[0] == len(current_utt)-1:
                    ## if only one satisfies and it is the last position where it is the ts
                    ## this is the only case it satisfies. add 1
                        self.threshold_results_total[threshold] += 1
                
                # ith_turn = 0              
                # while ith_turn < len(dialog_ts)-1:
                #     if ith_turn == 0:
                #         start_idx = 0
                #         end_idx = dialog_ts[ith_turn]
                #     else:
                #         start_idx = dialog_ts[ith_turn-1]+1
                #         end_idx = dialog_ts[ith_turn]
                #     trp_proj = out_trp[b].cpu()
                #     current_utt = trp_proj[start_idx:end_idx]
                #     idx_above_threshold = [i for i in range(len(current_utt)) if current_utt[i]>threshold]
                #     #print(idx_above_threshold)
                #     #pdb.set_trace()
                #     if len(idx_above_threshold)==1 and idx_above_threshold[0] == len(current_utt)-1:
                #     ## if only one satisfies and it is the last position where it is the ts
                #     ## this is the only case it satisfies. add 1
                #         self.threshold_results_total[threshold] += 1
                #     #max_trp_idx = torch.max(trp_proj[start_idx:end_idx],0).indices
                #     #max_trp = torch.max(trp_proj[start_idx:end_idx],0).values

                #     ith_turn += 1
        self.n_turns += out_trp.shape[0]
    def find_best_word_threshold(self, y_hat, y, interval=0.01):
        thresholds = np.arange(0, 1, interval)
        threshold_results = dict()
        for threshold in thresholds:
            threshold_results[threshold] = 0
        
        for threshold in thresholds:
            calculate_accuracy = Accuracy(num_classes=2,average="macro").to(y.device)
            y_hat_new = [0 if el<threshold else 1 for el in y_hat]
            y_hat_new = torch.tensor(y_hat_new).to(y.device)
            bacc = calculate_accuracy(y_hat_new.long(), y.long())
            threshold_results[threshold] = bacc
        
        #n_word_token = y.shape[-1]
        #self.total_word_token += n_word_token
        return threshold_results    
    def remove_last_sentence_and_pad(self,input_ids, input_ids_original, ts_idx_long):
            ## modified b2 0604. update 0606
            # 1. pad all complete last_sent (if truncated, padd too, and pad another last_sent)
            # 2. remove truncated last_sent in input_ids_original
            # (because sometimes the truncated sent only has 1 word)
        i = 0
        while i < len(input_ids):
            if input_ids[i][-1] not in [50256, 50257]: 
                
                # end with word token (not with <endoftext> or <ts>) 
                # ==> last sentence is truncated
                # remove in input_ids_original, so not to be taken for sent representation
                try:
                    last_ts_idx = ts_idx_long[i][-1]
                    pad_len = len(input_ids_original[i][last_ts_idx+1:])
                    pad_tensor = torch.Tensor([50256]).to(torch.int64).repeat(pad_len).to(self.device)
                    input_ids_original[i][last_ts_idx+1:] = pad_tensor
                except IndexError:
                    print('else; ts_idx[i]', ts_idx_long[i])
            
            ## after removing truncated sentence
            ## remove the last sentence (no next_sent available)
            try:
                second_last_ts_idx = ts_idx_long[i][-2]                   
                pad_len = len(input_ids[i][second_last_ts_idx+1:])
                pad_tensor = torch.Tensor([50256]).to(torch.int64).repeat(pad_len).to(self.device)
                input_ids[i][second_last_ts_idx+1:]= pad_tensor
            except IndexError:
                print('input_ids[i][-1] in [50256, 50257]; ts_idx[i]', ts_idx_long[i])
            i+=1        
        return input_ids, input_ids_original
    def discard_short_dialogs(self, input_ids, n_turns=1):
             # clean utterances, should be longer than 3 turns
            # b2 changed 0604: changed from 2 turns to 3 turns       
        ts_idx = [(sublist == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0] for sublist in input_ids]
        valid_input_ids_rows = [i for i in range(len(ts_idx)) if len(ts_idx[i])>n_turns]
        ts_idx_long = [ts_idx[idx] for idx in valid_input_ids_rows]
        input_ids = input_ids[valid_input_ids_rows]        
        
        return input_ids, ts_idx_long
    
    def get_future_utt_ppl (self, new_dialog, batch, loss_fn):
        turngpt_out = self.original_turngpt.string_list_to_trp_inputid(new_dialog)
        ts_idx_turngpt = [(sublist == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0] for sublist in turngpt_out['input_ids']]
        
        future_utt_trp_prob = turngpt_out['trp_probs'][:,ts_idx_turngpt[0][-2]+1:].squeeze(0)
        if turngpt_out['input_ids'][:,ts_idx_turngpt[0][-2]+1:].squeeze(0)[-1]==50256:
            pdb.set_trace()
        
        future_utt_trp_prob = future_utt_trp_prob[:-1]
        future_labels = torch.zeros(len(future_utt_trp_prob))
        future_labels[-1] =1
        loss = loss_fn(future_utt_trp_prob.to(self.device), future_labels.float().to(self.device))
        #pdb.set_trace()
        ppl = torch.exp(loss)
        return ppl
    def get_future_utt_ppl_gpt3 (self, new_dialog, batch, loss_fn):
        gpt3_probs = openai.Completion.create(model="text-davinci-002",
                                    prompt=new_dialog,
                                    temperature=0.0,
                                    max_tokens=0,
                                    logprobs=0,
                                    echo=True,)
        #pdb.set_trace()
        turngpt_out = self.original_turngpt.string_list_to_trp_inputid(new_dialog)
        ts_idx_turngpt = [(sublist == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0] for sublist in turngpt_out['input_ids']]
        
        future_utt_trp_prob = turngpt_out['trp_probs'][:,ts_idx_turngpt[0][-2]+1:].squeeze(0)
        if turngpt_out['input_ids'][:,ts_idx_turngpt[0][-2]+1:].squeeze(0)[-1]==50256:
            pdb.set_trace()
        
        future_utt_trp_prob = future_utt_trp_prob[:-1]
        future_labels = torch.zeros(len(future_utt_trp_prob))
        future_labels[-1] =1
        loss = loss_fn(future_utt_trp_prob.to(self.device), future_labels.float().to(self.device))
        #pdb.set_trace()
        ppl = torch.exp(loss)
        return ppl
    @torch.no_grad()
    def percent_future_attention (self, attns,input_ids):
        '''
        attns = [batch_size, n_head, len_dialog, len_dialog]
        '''
        future_attentions = [] # one value for each dialog (average future attention by head)
        future_attentions_all_heads = [] # n_head value for each dialog
        ts_idx = [(sublist == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0] for sublist in input_ids]
        n_heads = attns.shape[1]
        for i, attn in enumerate(attns):
            future_ts = ts_idx[i][0]
            attn_future_all = [sum(attns[i,j,-1,:future_ts]) for j in range(12)]
            future_attentions.append(sum(attn_future_all)/n_heads)
            future_attentions_all_heads.append(attn_future_all)       
        return torch.FloatTensor(future_attentions), torch.FloatTensor(future_attentions_all_heads), ts_idx
    
    @torch.no_grad()
    def calculate_main_attention (self, attns, future_attentions,ts_idx, percent = 0.9):
        '''
        attention: one per each converstation, only last one permutated
        [batch_size, n_head, len_dialog, len_dialog]
        '''
        furthest_indices_all = []
        rel_furthest_indices_all = []
        future_attentions_thresholds = future_attentions * percent
        for j, attn in enumerate(attns):
            future_ts = ts_idx[j][0]
            current_ts = ts_idx[j][-1]
            future_attn_threshold = future_attentions_thresholds[j]
            # average over attention heads
            # attn = [n_head, len_dialog, len_fut_utt]
            average_attn = torch.sum(attn[:,:,:future_ts],0)/attn.shape[0]
            # average over y axis
            # fut_attn_by_tok = one val for each tok in future utt
            #fut_attn_by_tok = sum(average_attn, 0)/average_attn.shape[0]
            fut_attn_by_tok = average_attn[current_ts]
            if sum(fut_attn_by_tok) < future_attn_threshold:
                # doesn't reach threshold at all
                furthest_indices_all.append(-1)
                rel_furthest_indices_all.append(-1)
            else: # reaches threshold at some point
                cumulative_attn = 0
                i = 0
                while i < len(fut_attn_by_tok) and cumulative_attn < future_attn_threshold:
                    cumulative_attn += fut_attn_by_tok[i]
                    if cumulative_attn > future_attn_threshold:
                        #print(cumulative_attn)
                        #print(future_attentions_thresholds)
                        #print('-----')
                        furthest_indices_all.append(i)
                        rel_furthest_indices_all.append(i/len(fut_attn_by_tok))
                        break
                    else:
                        i+=1
        return torch.FloatTensor(furthest_indices_all), torch.FloatTensor(rel_furthest_indices_all)
    def forward(
        self,
        input_ids=None,
        speaker_ids=None,
        labels=None,
        mc_labels=None,
        use_cache=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        """
        Simple rewrite of original:

            https://github.com/huggingface/transformers/blob/439a43b6b403205eeda2d62645fc16c93627d30d/src/transformers/models/gpt2/modeling_gpt2.py#L1086
        """
        return_dict = (
            return_dict
            if return_dict is not None
            else self.transformer.config.use_return_dict
        )
        #self.transformer.config.output_attentions = True
        
        
        #pdb.set_trace()
        ## added 0912 by b2 to match conditional, for remove last sentence in
        #input_ids, ts_idx_long = self.discard_short_dialogs(input_ids, 1)
        #input_ids_original = input_ids.clone()
        #input_ids, input_ids_original = self.remove_last_sentence_and_pad(input_ids, input_ids_original,ts_idx_long)
        #pdb.set_trace()
        #try:
        transformer_outputs = self.transformer.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=speaker_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        # except RuntimeError:
            
        #     print('runtime error, skip batch')
        #     #pdb.set_trace()
        #     return None
        # Set device for model parallelism
        if self.transformer.model_parallel:
            torch.cuda.set_device(self.transformer.transformer.first_device)
            hidden_states = hidden_states.to(self.transformer.lm_head.weight.device)

        # Language Modeling
        ## b2: this predicts the next token
        ##     we want to predict 1) turn shift or not and 2) the next utt embedding
        
        lm_logits = self.transformer.lm_head(hidden_states)
        lm_loss = None
        if labels is not None:
            ## 10/3 modify lm_labels, mask current and future tokens
            #pdb.set_trace()
            
            #future_token_idx = [(el == self.tokenizer.future_token_id).nonzero(as_tuple=True)[0] for el in speaker_ids]
            #indices_for_training = speaker_ids != self.tokenizer.future_token_id
            #labels = torch.masked_select(labels, indices_for_training)
            labels = self.mask_future_token(labels, speaker_ids)
            #labels = self.mask_current_token(input_ids, labels)
            labels = self.mask_context_token(input_ids, labels)
            lm_loss = self.cross_entropy_loss(lm_logits, labels)

        ## TODO:
        # 1) combine the sentence encoding model output with hidden_states
        #    -> binary classification
        
        ## get the next utterance in current batch
        ## format in the same shape as hiddent_states 
        
        ## call encoding_model
        # sent_encode = encoding_model
        # MultiTask Modeling
        mc_logits = None
        mc_loss = None
        if self.trp_projection_steps > 0:
            # NOTE:
            # Assumed to only guess a single class
            mc_logits = self.trp_projection_head(hidden_states).squeeze(-1)

            if mc_labels is not None:
                bad_mc_labels = torch.full(mc_labels.shape, -100,
                                           dtype = torch.float32).to(self.device)
                if not torch.equal(bad_mc_labels, mc_labels):
                    
                    mc_labels = self.mask_future_token(mc_labels, speaker_ids)
                    #mc_labels = self.mask_current_token(input_ids, mc_labels)
                    mc_labels = self.mask_context_token(input_ids, mc_labels)
                    mc_loss = self.bce_loss(mc_logits, mc_labels)
                else:
                    print('bad mc labels')
                    return None
        # if not return_dict:
        #     output = (lm_logits, mc_logits) + transformer_outputs[1:]
        #     if mc_loss is not None:
        #         output = (mc_loss,) + output
        #     return ((lm_loss,) + output) if lm_loss is not None else output
        #print('forward pass')
        #if lm_loss is not None:
        if torch.isnan(mc_loss):
            return None
        
        out = GPT2DoubleHeadsModelOutput(
            loss=lm_loss,
            mc_loss=mc_loss,
            logits=lm_logits,
            mc_logits=mc_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
        #pdb.set_trace()
        return out

    def configure_optimizers(self):
        # NOTE:
        # Use multiple optimizers for transformer and projection?
        # see:
        #   https://pytorch-lightning.readthedocs.io/en/stable/common/optimizers.html#use-multiple-optimizers-like-gans-manual
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def on_save_checkpoint(self, checkpoint):
        """We must save the tokenizer used during training"""
        checkpoint["tokenizer"] = self.tokenizer

    def on_load_checkpoint(self, checkpoint):
        """We must load the tokenizer used during training and resize the embeddings appropriately"""
        if "tokenizer" in checkpoint:
            print("#" * 70)
            print("LOAD CHECKPOINT TOKENIZER")
            self.tokenizer = checkpoint["tokenizer"]
            print("Loaded tokenizer")
            print(self.tokenizer)

            # Add extra embeddings for custom tokens
            self.transformer.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
            print("Resized weights")
            print("#" * 70)
            print(self.trp_projection_steps)
            #pdb.set_trace()
        #if 'trp_projection_head.weight' in checkpoint['state_dict']:
            
            #self.trp_projection_head = nn.Linear(self.transformer.config.hidden_size, 1)
    def training_step(self, batch, batch_idx):
        lm_labels = self.get_labels(batch["input_ids"], mask=batch["attention_mask"])
        #pdb.set_trace()
        proj_labels = None
        if self.trp_projection_steps > 0:
            proj_labels = self.get_projection_labels(
                batch["input_ids"], mask=batch["attention_mask"]
            )

        if self.omit_dialog_states:
            batch["speaker_ids"] = None

        out = self.forward(
            batch["input_ids"],
            speaker_ids=batch["speaker_ids"],
            labels=lm_labels,
            mc_labels=proj_labels,
        )
        #pdb.set_trace()
        if out is None:
            return None
        if self.trp_projection_steps > 0:
            self.log("loss_lm", out["loss"])
            self.log("loss_projection", out["mc_loss"])
            total_loss = out["loss"] + out["mc_loss"]
        else:
            self.log("loss", out["loss"])
            total_loss = out["loss"]
        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        lm_labels = self.get_labels(batch["input_ids"], mask=batch["attention_mask"])
        #pdb.set_trace()
        proj_labels = None
        if self.trp_projection_steps > 0:
            proj_labels = self.get_projection_labels(
                batch["input_ids"], mask=batch["attention_mask"]
            )
            if proj_labels is None:
                return None
        te = torch.full(proj_labels.shape, -100,
                        dtype = torch.float32)
        mc_labels = proj_labels.to(te.device)
        
        if torch.equal(te,mc_labels):
            print('same')
            pdb.set_trace()
            return None
        if self.omit_dialog_states:
            batch["speaker_ids"] = None
        
        # enable attention during validation
        self.transformer.config.output_attentions = True
        #pdb.set_trace()
        out = self.forward(
            batch["input_ids"],
            speaker_ids=batch["speaker_ids"],
            labels=lm_labels,
            mc_labels=proj_labels,
        )
        if out is None:
            return None

        if self.trp_projection_steps > 0:
            self.log("val_loss_lm", out["loss"])
            self.log("val_loss_projection", out["mc_loss"])
            total_loss = out["loss"] + out["mc_loss"]
        else:
            total_loss = out["loss"]

        self.log("val_loss", total_loss, on_step = True)
        
        #if out['mc_loss'] > 0.1047*2:
            #pdb.set_trace()
        # y_hat, y = self.get_trp_proj_word_with_tgt(out["mc_logits"], proj_labels)
        # if self.y is None:
        #     self.y = y
        # else: 
        #     self.y = torch.cat((self.y, y), 0 )
        
        # if self.y_hat is None:
        #     self.y_hat = y_hat
        # else:
        #     self.y_hat = torch.cat((self.y_hat, y_hat),0)
        
        
        if self.calculate_metric:
            self.find_best_threshold_turnlevel_future(out, batch["input_ids"])
            from collections import Counter 
            #pdb.set_trace()
            #self.threshold_results_total = dict(Counter(self.threshold_results_total) +
            #                               Counter(threshold_results))
            
            turnlevel_max_at_end = self.get_turnlevel_acc_max_at_end_future(out, batch["input_ids"])
            self.turnlevel_max += turnlevel_max_at_end
        if torch.isnan(out['mc_loss']):
            pdb.set_trace()
            return None
        #print('self.calculate_ppl: ', self.calculate_ppl)
        if self.calculate_ppl:
            #with torch.no_grad():
            loss_fn = nn.BCEWithLogitsLoss()
            threshold = 0.39
            out["probs"] = out["logits"].softmax(dim=-1)
            out["trp_probs"] = self.get_trp(out["probs"])
            if "mc_logits" in out:
                out["trp_proj"] = out["mc_logits"].sigmoid()
                
            if self.original_turngpt is not None:
                self.original_turngpt.eval()
                print('loaded original turngpt')
            
            ts_idx = [(sublist == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0] for sublist in batch['input_ids']]
            for idx, sublist_ts_idx in enumerate(ts_idx):
                #print('idx')
                #print('start for loop: ', idx)
                current_utt_idx_start = sublist_ts_idx[-2]
                current_utt_prob = out['trp_probs'][idx][current_utt_idx_start+1:]
                # current_utt_ids = batch['input_ids'][idx][current_utt_idx_start+1:]
                #pdb.set_trace()
                #idx_above_threshold = (current_utt_prob > threshold).nonzero(as_tuple=True)
                idx_above_threshold = [i for i in range(len(current_utt_prob)) if current_utt_prob[i]>threshold]
                
                #pdb.set_trace()
                #pred_turnshift_idx = next(x[0] for x in enumerate(current_utt_prob) if x[1] > threshold)
                if idx_above_threshold == []:
                    # doesn't predict ts for this utterance, skip
                    self.num_below_threshold +=1
                    #print('?')
                    continue
                    
                else:    
                    #pdb.set_trace()
                    pred_turnshift_idx = idx_above_threshold[0]
                    #print('above threshold: ', idx)
                    #pred_utt_ids = current_utt_ids[:pred_turnshift_idx+1]
                    #pred_utt_ids.append(model.tokenizer.eos_token_id)
                    future_utt_ids = batch['input_ids'][idx][:sublist_ts_idx[0]+1]
                    context_curr_utt_ids = batch['input_ids'][idx][sublist_ts_idx[0]+1:sublist_ts_idx[-2]+1+pred_turnshift_idx+1]
                    try: 
                        last_tok = context_curr_utt_ids[-1]
                    except IndexError:
                        pdb.set_trace()
                    if last_tok != self.tokenizer.eos_token_id:
                        #just insert <ts> token if there isn't one
                        context_curr_utt_ids = torch.cat(
                            (context_curr_utt_ids, torch.LongTensor([self.tokenizer.eos_token_id]).to(self.device)), 0
                        )
                        
                    new_dialog_ids = torch.cat((context_curr_utt_ids, future_utt_ids), 0) # add future utt back
                    new_dialog = self.tokenizer.decode(new_dialog_ids)
                    
                    ## use TURNGPT to evaluate
                    ppl = self.get_future_utt_ppl(new_dialog,batch, loss_fn)
                    #ppl = self.get_future_utt_ppl_gpt3(new_dialog,batch, loss_fn)
                    self.future_utt_ppl+= ppl
                    self.num_future_utts +=1
        attns = out['attentions'][-1]
        fut_attns, fut_attns_heads, ts_idx = self.percent_future_attention(attns, batch['input_ids'])    
        furthestidx, relidx = self.calculate_main_attention(attns, fut_attns, ts_idx)
        
        #self.n_turns += fut_attns.shape[0]
        if self.furthest_idx_all is None:
            self.furthest_idx_all = furthestidx
        else:
            self.furthest_idx_all = torch.cat(
                [self.furthest_idx_all, furthestidx], dim = 0
            )
        if self.rel_furthest_idx_all is None:
            self.rel_furthest_idx_all = relidx
        else:
            self.rel_furthest_idx_all = torch.cat(
                [self.rel_furthest_idx_all, relidx], dim = 0
            )
        if self.precent_future_attention is None:
            self.precent_future_attention = fut_attns
        else:
            self.precent_future_attention = torch.cat(
                [self.precent_future_attention, fut_attns], dim = 0
            )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Specify the hyperparams for this LightningModule"""
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, conflict_handler="resolve"
        )
        parser.add_argument("--pretrained_model_name_or_path", type=str, default="gpt2")
        parser.add_argument(
            "--pretrained",
            type=bool,
            default=True,
            help="Load pretrained weights or not.",
        )

        # Model specific
        parser.add_argument("--embd_pdrob", type=float, default=None)
        parser.add_argument("--attn_pdrob", type=float, default=None)
        parser.add_argument("--resid_pdrob", type=float, default=None)
        parser.add_argument("--n_head", type=int, default=None)
        parser.add_argument("--n_layer", type=int, default=None)
        parser.add_argument("--n_embd", type=int, default=None)
        parser.add_argument("--activation_function", type=str, default=None)

        # TurnGPT specific
        parser.add_argument(
            "--omit_dialog_states",
            action="store_true",
            help="To omit dialog-states in transformer embedding",
        )
        parser.add_argument("--trp_projection_steps", default=-1, type=int)
        parser.add_argument(
            "--no_train_first_n",
            default=-1,
            type=int,
            help="Don't train on the n first tokens.",
        )
        parser.add_argument(
            "--trp_projection_type",
            default="linear",
            type=str,
            help="'Linear' or 'Attention'",
        )

        # Training
        parser.add_argument(
            "--dropout",
            default=None,
            type=float,
            help="Set to None which uses the values in the original config.json",
        )
        parser.add_argument("--learning_rate", default=6.25e-5, type=float)
        parser.add_argument("--weight_loss", action="store_true")
        parser.add_argument("--weight_eos_token", type=float, default=1.0)
        parser.add_argument("--weight_regular_token", type=float, default=0.5)
        parser.add_argument("--calculate_metric", action='store_true', help='calculate acc, only use when run validation alone')
        parser.add_argument("--calculate_ppl", action='store_true', help='calculate ppl, only use when run validation alone')
        parser.add_argument("--trp_eval_type", type=str, help="choose between: trp_proj , trp_probs")
        return parser


if __name__ == "__main__":
    from os.path import join

    parser = ArgumentParser()
    parser = TurnGPT.add_model_specific_args(parser)
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    # Fresh Training
    #fresh = False
    fresh = False # by b2
    if fresh:
        model = TurnGPT(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            trp_projection_steps=args.trp_projection_steps,
            trp_projection_type=args.trp_projection_type,
            weight_loss=args.weight_loss,
            weight_eos_token=args.weight_eos_token,
            weight_regular_token=args.weight_regular_token,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            pretrained=args.pretrained,
            no_train_first_n=args.no_train_first_n,
            omit_dialog_states=args.omit_dialog_states,
        )
        model.init_tokenizer()
        model.initialize_special_embeddings()
    else:
        # projection
        # chpt = join(
        #     "assets/TurnGPT_proj/version_0/checkpoints/epoch=45-val_loss=-3.37196.ckpt"
        # )
        # # only LM
        # chpt = join(
        #     "assets/TurnGPT/version_0/checkpoints/epoch=11-val_loss=1.23640.ckpt"
        # )
        chpt = join("runs/TurnGPT/TurnGPT_rnj0r9nj/epoch=9_val_loss=1.6756.ckpt")
        model = TurnGPT.load_from_checkpoint(chpt).to("cuda")

    # turn_list = [
    #     "Hello there I basically had the worst day of my life",
    #     "Oh no, what happened?",
    #     "Do you want the long or the short story?",
    # ]
    turn_list = [
        ['What did you do yesterday?', 'I went hiking with my friends John and Mary. What did you do?', 'I was painting the wall of my garage.'],
        ['What did you do yesterday?', 'I went hiking with my friends John and Mary.', 'That sounds cool!'],
        ['What did you do yesterday?', 'I went hiking with my friends John and Mary. What did you do?', 'That sounds cool!'],
        ['What did you do yesterday?', 'I went hiking with my friends John and Mary. What did you do?', 'Are you okay?'],
        ['What did you order?', 'What we always have for brunch here, tuna sandwiches, fries, pudding, and coffee.', 'That sounds good!'],
        ['What did you order?', 'What we always have for brunch here, tuna sandwiches, fries, pudding, and coffee.', 'Are you okay?'],
        ['What did you order?', 'What we always have for brunch here, do you want anything else today?', 'No, that sounds good!']
    ]

    # gen = generate(
    #     model,
    #     context=turn_list,
    #     n_steps=200,
    #     top_p=0.9,
    #     top_k=-1,
    #     n_trajectories=20,
    #     strategy="sample",
    #     stop_at_eos=True,
    # )
    # remove duplicates
    # l = (gen["input_ids"][0] != -1).sum()
    # G = {"tokens": [gen["tokens"][0]], "probs": [gen["probs"][0][:l].cpu()]}
    # for i, g in enumerate(gen["tokens"][1:]):
    #     if g not in G["tokens"]:
    #         l = (gen["input_ids"][i] != -1).sum()
    #         G["tokens"].append(g)
    #         G["probs"].append(gen["probs"][i][:l].cpu())

    #########################################################
    # turn_list = [
    #     [
    #         "Hello there I basically had the worst day of my life",
    #         "Oh no, what happened?",
    #         "Do you want the long or the short story?",
    #     ],
    #     ["yesterday we met in the park", "okay when will you meet again", "tomorrow"],
    # ]
    out = model.string_list_to_trp(turn_list)
    # for k, v in out.items():
    #     print(f"{k}: {v}")

    for b in range(out["trp_probs"].shape[0]):
        proj = out["trp_proj"][b].cpu() if "trp_proj" in out else None
        fig, ax = plot_trp(
            trp=out["trp_probs"][b].cpu(),
            proj=proj,
            text=out["tokens"][b],
            unk_token=model.tokenizer._tokenizer.unk_token,
            plot=True,
        )
    pdb.set_trace()