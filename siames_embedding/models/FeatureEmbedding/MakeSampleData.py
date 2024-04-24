import random
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_onehot_data(length, *args) :
    return [random.randint(*args[0]) for _ in range(length)]

def make_linear_data(length, *args) :
    return [random.random() * args[0] for _ in range(length)]

def make_multihot(length, *args) :
    args = args[0]
    multihot_length = args[0]
    min_range = args[1]
    max_range = args[2]
    
    return [[random.randint(min_range, max_range) for _ in range(multihot_length)] for _ in range(length)]

def random_config(feature_type) :

    if feature_type == "onehot" :
        val1 = random.randint(1, 30)
        val2 = random.randint(1, 30)
        return min(val1, val2), max(val1, val2)
        
    if feature_type == "linear" :
        return random.randint(1, 10)
    
    if feature_type == "multihot" :
        val1 = random.randint(1, 30)
        val2 = random.randint(1, 30)
        return random.randint(1, 10), min(val1, val2), max(val1, val2)
    
    return

def make_independent_data(datamap, length) :

    data = {}
    
    for k, v in datamap.items() :
        if v == "onehot" :
            data[k] = make_onehot_data(length, random_config(v))
        if v == "linear" :
            
            data[k] = make_linear_data(length, random_config(v))
        if v == "multihot" :
            data[k] = make_multihot(length, random_config(v))
    
    return pd.DataFrame(data)


def make_dataset(target_variable, datamap, length) :
    
    return pd.concat([make_independent_data(datamap, length).assign(y  = i) for i in range(target_variable)])