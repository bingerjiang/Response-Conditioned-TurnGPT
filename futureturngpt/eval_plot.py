import os
import openai
from futureturngpt.model import TurnGPT, TurnGPTWandbCallbacks
from os.path import join
import re
import pdb
from futureturngpt.plot_utils import plot_trp
import wandb
import matplotlib.pyplot as plt
import torch
model_type = 'futureturngpt'

future_turngpt = True
if future_turngpt:
    model_folder = 'future'
else:
    model_folder = 'turngpt'
if future_turngpt:
    #chpt_path = 'runs/futureTurnGPT/futureTurnGPT_3qrpr59m_epoch=3_val_loss=0.2666.ckpt' # no sod token
    chpt_path = 'runs/futureTurnGPT/futureTurnGPT_3sdnm1mo/epoch=3_val_loss=0.2632.ckpt'
    chpt = join(
            chpt_path
        )
    model = TurnGPT.load_from_checkpoint(chpt)
else:
    from original_turngpt.model import originalTurnGPT, originalTurnGPTWandbCallbacks
    chpt = join(
        "/home/binger/repos/TurnGPT/runs/TurnGPT/TurnGPT_2cr6pudn/epoch=10_val_loss=1.7908.ckpt"
    )
    model =  originalTurnGPT.load_from_checkpoint(chpt)
turn_list = [
        #['What did you do yesterday?', 'I went hiking with my friends John and Mary. What did you do yesterday?', 'I was painting the wall of my garage.'],
        #['What did you do yesterday?', 'I went hiking with my friends John and Mary.', 'That sounds cool!'],
        #['What did you do yesterday?', 'I went hiking with my friends John and Mary. What did you do yesterday?', 'That sounds cool!'],
        #['What did you do yesterday?', 'I went hiking with my friends John and Mary. What did you do yesterday?', 'Are you okay?'],
        #['What did you order?', 'What we always have for brunch here, tuna sandwiches, fries, pudding, and coffee.', 'That sounds good!'],
        #['What did you order?', 'What we always have for brunch here, tuna sandwiches, fries, pudding, and coffee.', 'Are you okay?'],
        #['What did you order?', 'What we always have for brunch here, do you want anything else today?', 'No, that sounds good!']
        #['Did you and Mary meet yesterday?','Yes, we met in the park.',"That's great, I'm glad to hear that! when will you meet again?",'tomorrow.']
        ['Yesterday we met in the park.','Okay, when will you meet again?','tomorrow.']
        #["peter enough with your computer games go do your homework now","can't i play more?","no, stop playing computer games", "mom i'll be finished soon","peter, if you don't turn off your computer then i won't allow you to play it again starting next week."]
        #["hi brittany, what are you doing with all of your clothes on your bed?","i'm trying to decide what to wear to school the first day","oh mom didn't tell you?"," didn't tell me what?"," this school you're going to is going to make your life easy"," what are you talking about brother spill it"," uniforms sis, no more worrying about appearances","you mean i have to wear the same thing every day mom"]
        #["hello sir i'm ready for you","is it my turn?","yes, please sit on the chair. how do you want to have your hair cut?","not too long, cut a little off behind and on both sides too","ok, now lean back a little and keep still. i'm going to shave your face."]
    ]
turn_list = [
    ["Hi, I'd like to book an appointment with Dr. X.",
     "I'm sorry, Dr. X isn't available this month. Would you like to book an appointment with Dr. Y or Dr. Z instead?",
     "Are there any other doctors available this month?"
    ],
    ["Hi, I'd like to book an appointment with Dr. X.",
     "I'm sorry, Dr. X isn't available this month. Would you like to book an appointment with Dr. Y or Dr. Z instead?",
     "I'd like to book an appointment with Dr. Y then, thank you."
    ],
]
model.transformer.config.output_attentions = True
out = model.string_list_to_trp(turn_list)

plot = True
if plot:
    for b in range(out["trp_probs"].shape[0]):
        proj = out["trp_proj"][b].cpu() if "trp_proj" in out else None
        #pdb.set_trace()
        text = out['tokens'][b]
        #pdb.set_trace()
        #text.insert(0, '<nextutt>')
        #text.pop()
        fig, _ = plot_trp(
            trp=out["trp_probs"][b].cpu(),
            proj=proj,
            text=text,
            unk_token=model.tokenizer.unk_token,
            eos_token=model.tokenizer.eos_token,
            plot=True,
        )
        #pdb.set_trace()
        fig.savefig('plots/'+model_folder+f'/1227_{b}')


#out = model.string_list_to_trp(turn_list)
pdb.set_trace()

attn_all = out['attentions']
attn = attn_all[-1]
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
        if t == model.tokenizer.unk_token:
            max_idx = n
            break
    text = text[:max_idx]
    attn = attn[:max_idx]
    #pdb.set_trace()
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
        #fig.savefig(f'attn_plots/1212_{b}.png')
    x = torch.arange(len(text))
    ax[b].set_xticks(x)
    ax[b].set_yticks(x)

    plt.tight_layout()
    if text is not None:
        ax[b].set_xticklabels(text, rotation=60)
        ax[b].set_yticklabels(text, rotation=60)
    fig.savefig('attn_plots/1212_'+model_folder+f'{b}.png')
    plt.close("all")
pdb.set_trace()