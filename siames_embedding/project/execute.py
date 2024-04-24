#!/usr/bin/env python
# coding: utf-8

%load_ext autoreload
%autoreload 2

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd

sys.path.append("../")
from models.FeatureEmbedding.Model import EmbeddingModel, SiameseNetwork
from models.FeatureEmbedding.CustomDataset import SiamesDataset
from models.FeatureEmbedding.MakeSampleData import make_dataset



class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

def search_num_sparse_features(df, datamap, y_col = "y") :
    datamap = datamap.copy()
    datamap.pop("y")
    return max(*[df[k].apply(max).max() if v == "multihot" else df[k].max() for k, v in datamap.items() if v in ["multihot", "onehot"]])


datamap1 = {"a" : "linear", "b" : "multihot", "c" : "onehot", "d" : "multihot", "y" : "onehot"}
datamap2 = {"a" : "linear", "b" : "multihot", "c" : "onehot", "y" : "onehot"}

data1 = make_dataset(2, datamap1, 1000)
data2 = make_dataset(2, datamap2, 1000)

dataset = SiamesDataset(data1, datamap1, data2, datamap2, data1, "y")
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 모델 인스턴스 생성
num_sparse_features = max(search_num_sparse_features(data1, datamap1), search_num_sparse_features(data2, datamap2)) + 2  # Embedding Idx의 최대값 + 1
num_linear_features = 1  # linear feature의 개수 - 개별로 넣으면 1, 묶음이면 len(feature)
embedding_dim = 100
hidden_dim = 20
output_dim = 10  # 예측할 출력의 차원 (예: 회귀의 경우 1, 이진 분류의 경우 1)

model_config = {
    "num_sparse_features" : num_sparse_features,
    "search_num_sparse_features" : search_num_sparse_features,
    "num_linear_features" : num_linear_features,
    "embedding_dim" : embedding_dim,
    "hidden_dim" : hidden_dim,
    "output_dim" : output_dim,
}

model1 = EmbeddingModel(datamap1, model_config = model_config)
model2 = EmbeddingModel(datamap2, model_config = model_config)

siam_model = SiameseNetwork(model1, model2)
print(siam_model)

criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(siam_model.parameters(),lr = 0.0005)

counter = []
loss_history = [] 
iteration_number= 0


torch.sum(siam_model.model1.b(dataset.__getitem__(0)[0]["b"]), dim = 1)

for epoch in range(0, 100):
    for i, data in enumerate(dataloader):
        x1, x2, y = data
        
        optimizer.zero_grad()
        output1, output2 = siam_model(x1, x2)
        
        loss_contrastive = criterion(output1,output2,y)
        loss_contrastive.backward()
        optimizer.step()
    if epoch %10 == 0 :
        print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
        iteration_number +=10
        counter.append(iteration_number)
        loss_history.append(loss_contrastive.item())



torch.sum(siam_model.model1.b(dataset.__getitem__(0)[0]["b"]), dim = 1)




