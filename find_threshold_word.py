import pickle
from torchmetrics.classification import Accuracy
import numpy as np
import torch
def find_best_word_threshold( y_hat, y, interval=0.01):
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
with open('results.pkl','rb') as f:
    myl=pickle.load(f)

results = find_best_word_threshold(myl[0],myl[1])
import pdb
pdb.set_trace()