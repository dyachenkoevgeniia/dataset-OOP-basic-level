import pandas as pd
from pathlib import Path
import numpy as np
from dataloader import DataLoader
from visualizer import Visualizer

class Saveable:
    def to_csv(self, path, **kwargs):
        self.data.to_csv(path, index=False, **kwargs)

    def to_json(self, path, **kwargs):
        self.data.to_json(path, orient="records", **kwargs)

    def to_excel(self, path, **kwargs):
        self.data.to_excel(path, index=False, **kwargs)

class Dataset(Saveable):
    instances=0
    def __init__(self, data):
        self.data=data
        self.visualizer= Visualizer(self.data)
        Dataset.instances+=1
    
    def __str__(self):
        return f"Number of rows: {self.data.shape[0]} \nNumber of columns: {self.data.shape[1]} "

    def __len__(self):
        return len(self.data)

    def __eq__(self, other):
        return (self.data.shape == other.data.shape ) and (list(self.data.columns)==list(self.data.columns))

    def __getitem__(self,key):
        return self.data[key]

    @property
    def data_columns(self):
        return self.data.columns
    
    @classmethod
    def from_file(cls, path):
        ext=Path(path).suffix[1:]
        type_loader=DataLoader.loaders[ext]
        if type_loader is None:
            raise ValueError('Unsuported format')
        loader=type_loader(path)
        data=loader.load()
        return cls(data)


    class _Splitter:
        def __init__(self,data):
            self.data=data

        def split(self, ratio, seed=None):
            if seed is not None:
                np.random.seed(seed)

            n = len(self.data)
            indices=np.arange(n)
            np.random.shuffle(indices)

            train_idx= indices[:int(n*ratio)]
            test_idx= indices[int(n*ratio):]

            if hasattr(self.data, 'iloc'):
                train=self.data.iloc[train_idx]
                test=self.data.iloc[test_idx]
            elif isinstance(self.data, np.ndarray): 
                train = self.data[train_idx]
                test = self.data[test_idx]
            elif isinstance(self.data, list):
                train = [self.data[i] for i in train_idx]
                test = [self.data[i] for i in test_idx]
            else:
                raise TypeError('Unsupported data type')
            
            return train, test
        
        def split_xy(self, X, y, ratio, seed=None):
            if seed:
                np.random.seed(seed)

            n = len(X)
            indices = np.arange(n)
            np.random.shuffle(indices)

            split_point = int(n * ratio)
            train_idx = indices[:split_point]
            test_idx = indices[split_point:]

            if hasattr(X, 'iloc'):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            else:  
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

            return X_train, X_test, y_train, y_test
        
    def split(self, ratio, seed=None):
        return Dataset._Splitter(self.data).split(ratio, seed)
    
    def split_xy(self, X, y, ratio, seed=None):
        return Dataset._Splitter(self.data).split_xy(X,y,ratio,seed)
    


