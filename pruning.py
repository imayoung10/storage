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



def reshape_layer(next_layer_tensor, remaining_idx, prev_size):
    weight = next_layer_tensor.data

    if isinstance(remaining_idx, int):
        remaining_idx = [remaining_idx]  
    elif isinstance(remaining_idx, torch.Tensor):
        remaining_idx = remaining_idx.tolist() 


    if len(weight.shape) > 1:
        in_features = weight.shape[1]  
    else:
        in_features = weight.shape[0]  
   

   
    remaining_idx = [idx for idx in remaining_idx if idx < in_features]

 
    if len(remaining_idx) >= in_features:
        return next_layer_tensor

    
    if len(weight.shape) > 1:  
        reshaped_tensor = weight[:, remaining_idx, :, :]
    else:  
        reshaped_tensor = weight[remaining_idx]
    return reshaped_tensor

new_state_dict = state_dict
key_list = list(state_dict.keys())
new_state_dict = OrderedDict()
cnt = 0
no_prune = ['running_mean', 'running_var' , 'num_batches_tracked']
while cnt < len(key_list):
    key = key_list[cnt]

    if 'conv' in key and len(state_dict[key].shape) > 3:
        pruned_weight, remaining_idx = prune(state_dict[key], 0.3)  # Prune the layer weights


        new_state_dict[key] = pruned_weight

        while cnt + 1 < len(key_list) and  not any(item in key_list[cnt] for item in no_prune) and '24' not in key_list[cnt + 1]:
            cnt += 1
            print('before', state_dict[key_list[cnt]].data.shape)
            reshaped_tensor = reshape_layer(state_dict[key_list[cnt]], remaining_idx, pruned_weight.shape)
            new_state_dict[key_list[cnt]] = reshaped_tensor
            print('after ', new_state_dict[key_list[cnt]].data.shape)

        is_last = '24' in key_list[cnt]
        if is_last:
            break  
    else:

        new_state_dict[key] = state_dict[key]

    cnt += 1
torch.save(new_state_dict, 'pruned.pt')
