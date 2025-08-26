from abc import ABC, abstractmethod
import pandas as pd


class DataLoader(ABC):
    loaders = {}

    @abstractmethod
    def load(self):
        pass

    @classmethod
    def register(cls, ext, loader_cls):
        cls.loaders[ext]=loader_cls


class CSVLoader(DataLoader):
    def __init__(self,path,sep=',',encoding='utf-8'):
        self.path=path
        self.sep=sep
        self.encoding=encoding

    def load(self):
        return pd.read_csv(self.path, sep=self.sep, encoding=self.encoding)
    
DataLoader.register('csv', CSVLoader)

class JSONLoader(DataLoader):
    def __init__(self,path):
        self.path=path

    def load(self):
        return pd.read_json(self.path)
    
DataLoader.register('json', JSONLoader)
    
class ExcelLoader(DataLoader):
    def __init__(self,path,sheet_name=None):
        self.path=path
        self.sheet_name=sheet_name

    def load(self):
        return pd.read_excel(self.path, sheet_name=self.sheet_name)
    
DataLoader.register('xlsx', ExcelLoader)

