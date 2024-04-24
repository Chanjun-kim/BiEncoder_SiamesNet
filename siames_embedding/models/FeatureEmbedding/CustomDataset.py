import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class EmbeddingDataset(Dataset): 
    def __init__(self, df, datamap, y_col):
        
        self.x_data = {}
        for col, v in datamap.items() :
            if v == "linear" :
                self.x_data[col] = torch.FloatTensor(df[col].values).unsqueeze(1)
            if v == "multihot" :
                df = df.assign(**{col : lambda x : self._pad_sequence(x[col])})
                self.x_data[col] = torch.LongTensor(df[col].to_list())
            if v == "onehot" :
                self.x_data[col] = torch.LongTensor(df[col].values)
        
        self.y_data = torch.LongTensor(df[y_col].values)

  # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.y_data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        x = {x : y[idx] for x, y in self.x_data.items()}
        y = self.y_data[idx]
        return x, y
    
    def _get_max_multihot_size(self, series):
        return series.apply(len).max()

    def _get_max_multihot_value(self, series):
        return series.apply(max).max()
        
    def _pad_infinite(self, iterable, padding=None):
        from itertools import chain, repeat, islice
        return chain(iterable, repeat(padding))
    
    def _pad(self, iterable, size, padding=None):
        return list(islice(pad_infinite(iterable, padding), size))
        
    def _pad_sequence(self, series) :
        l = self._get_max_multihot_size(series)
        m = self._get_max_multihot_value(series)
        return series.apply(lambda x : self._pad(x, l, m + 1))


class SiamesDataset(Dataset): 
    def __init__(self, df1, datamap1, df2, datamap2, df_y, y_col = "y"):
        
        self.x_data1 = self._get_x_data(df1, datamap1)
        self.x_data2 = self._get_x_data(df2, datamap2)
        self.y_data = torch.LongTensor(df_y[y_col].values)

  # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.y_data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        x1 = {x : y[idx] for x, y in self.x_data1.items()}
        x2 = {x : y[idx] for x, y in self.x_data2.items()}
        y = self.y_data[idx]
        return x1, x2, y

    def _get_x_data(self, df, datamap) :
        
        x_data = {}
        for col, v in datamap.items() :
            if v == "linear" :
                x_data[col] = torch.FloatTensor(df[col].values).unsqueeze(1)
            if v == "multihot" :
                df = df.assign(**{col : lambda x : self._pad_sequence(x[col])})
                x_data[col] = torch.LongTensor(df[col].to_list())
            if v == "onehot" :
                x_data[col] = torch.LongTensor(df[col].values)

        return x_data
    
    def _get_max_multihot_size(self, series):
        return series.apply(len).max()

    def _get_max_multihot_value(self, series):
        return series.apply(max).max()
        
    def _pad_infinite(self, iterable, padding=None):
        from itertools import chain, repeat, islice
        return chain(iterable, repeat(padding))
    
    def _pad(self, iterable, size, padding=None):
        from itertools import chain, repeat, islice
        return list(islice(self._pad_infinite(iterable, padding), size))
        
    def _pad_sequence(self, series) :
        l = self._get_max_multihot_size(series)
        m = self._get_max_multihot_value(series)
        return series.apply(lambda x : self._pad(x, l, m + 1))