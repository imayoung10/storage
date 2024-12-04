import os
import json
import torch
import torch.nn as nn 
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import yolov6

model = torch.load('yolov5s.pt')
model = model['model']
state_dict = model.state_dict()

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def find(module, name=""):
    if "ConvBNSiLU" in name:
        print(name)
        return {name: module}
    res = {}
    for name1, module in model.named_modules():
        res.update(find_layers(
            module, name=name + '.' + name1 if name != '' else name1
        ))

res = find(model, "")
def prune(weight, rate):
    W = weight.data
    o = W.shape[0]
    num = int(o * rate)
    l2_norms = torch.sqrt(torch.sum(W ** 2, dim=(1, 2, 3)))
    _, top_idx = torch.topk(l2_norms, num, dim=0, largest=False)
    remaining_idx = torch.arange(W.size(0), device=W.device)
    remaining_idx = remaining_idx[~torch.isin(remaining_idx, top_idx)]
    if len(top_idx) == o:
        top_idx = top_idx[:-2]
    new_W = W[remaining_idx, :, :, :]
    print(f'before {W.shape[0]} after {new_W.shape[0]}')
    return new_W, remaining_idx



def reshape_layer(next_layer, remaining_idx, prev_size):

    weight = next_layer.data
    if len(remaining_idx) >= weight.shape[1]:
        return next_layer
    weight = weight[:, remaining_idx, :, :]
    next_layer.in_channels = prev_size[0]
    
    return next_layer

new_state_dict = state_dict
key_list = list(state_dict.keys())
cnt = 0
no_prune = ['act', 'bn']
while cnt < len(key_list):
    key = key_list[cnt]
    if 'conv' in key and len(state_dict[key].shape) >3:
        pruned_weight, remaining_idx = prune(state_dict[key], 0.3)
        cnt += 1
        while any(item in key_list[cnt] for item in no_prune) and '24' not in key_list[cnt]:
            cnt += 1
        is_last = '24' in key_list[cnt]
        new_state_dict[key] = reshape_layer(state_dict[key_list[cnt]], remaining_idx, pruned_weight.shape)
        if is_last:
            cnt = len(key_list)+100
    else:
        cnt += 1
torch.save(new_state_dict, 'pruned.pt')
