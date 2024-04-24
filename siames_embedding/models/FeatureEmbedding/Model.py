import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class EmbeddingModel(nn.Module):
    def __init__(self, datamap, model_config, y_col = "y"):
        super(EmbeddingModel, self).__init__()
         
        self.datamap = datamap
        self.y_col = y_col

        num_sparse_features = model_config["num_sparse_features"]
        search_num_sparse_features = model_config["search_num_sparse_features"]
        num_linear_features = model_config["num_linear_features"]
        embedding_dim = model_config["embedding_dim"]
        hidden_dim = model_config["hidden_dim"]
        output_dim = model_config["output_dim"]

        # 임베딩 레이어 초기화
        for k, v in datamap.items() :
            if v == "linear" :
                setattr(self, k, nn.Linear(num_linear_features, embedding_dim))
            elif v == "onehot" :
                setattr(self, k, nn.Embedding(num_sparse_features, embedding_dim))
            elif v == "multihot" :
                setattr(self, k, nn.Embedding(num_sparse_features, embedding_dim))
        fc1_embedding_dim = len(datamap) - 1
        # 다층 퍼셉트론(MLP) 레이어 초기화 
        self.fc1 = nn.Linear(embedding_dim * fc1_embedding_dim, hidden_dim)  # 임베딩된 특성이 3개이므로 *3
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def _get_multihot_embedding(self, embedding, method_multihot) :

        if method_multihot == "mean" :
            # 멀티-핫 특성의 임베딩 평균
            multihot_embedding = lambda x : torch.mean(embedding(x), dim = 1)
        elif method_multihot == "sum" :
            # 각 인덱스에 대한 임베딩 벡터를 더하여 합산
            multihot_embedding = lambda x : torch.sum(embedding(x), dim = 1)
            # multi_hot_embedded = torch.sum(embedding(x[k]), dim=1)
        elif method_multihot == "weighted_mean" :
            # 각 인덱스에 대한 임베딩 벡터를 가져오고 가중 평균 계산
            weights = torch.ones_like(emb_vectors)  # 간단히 모든 값에 대해 동일한 가중치 사용
            multihot_embedding = lambda x : torch.mean(embedding(x) * weights, dim=1)

        return multihot_embedding
        
    
    def forward(self, x, method_multihot = "sum"):
        # sparse feature 임베딩
        embedded = {}
        for k, v in self.datamap.items() :
            if k == self.y_col :
                continue

            embedding = getattr(self, k)

            if v == "multihot" :
                embedding = self._get_multihot_embedding(embedding, method_multihot)

            embedding_value = embedding(x[k])
            embedded[k] = embedding_value

        # 모든 임베딩된 특성을 결합
        combined_features = torch.cat([v for k, v in embedded.items()], dim=1)  # dim=1은 각 임베딩을 행 방향으로 결합
        
        # MLP 레이어 적용
        x = F.relu(self.fc1(combined_features))
        x = self.fc2(x)
        
        return x

# # Siames Network
class SiameseNetwork(nn.Module):
    def __init__(self, model1, model2):
        super(SiameseNetwork, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, input1, input2):
        output1 = self.model1(input1)
        output2 = self.model2(input2)
        return output1, output2