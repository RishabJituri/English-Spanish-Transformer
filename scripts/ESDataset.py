import torch
from torch.utils.data import Dataset
import pandas as pd
import ast 

class EnglishSpanishDataset(Dataset):
    def __init__(self, statfile, datasetpath):
        stats = open(statfile, mode="r")
        self.en_max = int(stats.readline())
        self.sp_max = int(stats.readline())
        
        self.dataset = pd.read_csv(datasetpath)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        row = self.dataset.iloc[idx]
        englishlist = row["english"]
        spanishlist = row["spanish"]
        
        englishfiller = torch.zeros([self.en_max - len(englishlist)])
        input = torch.cat([torch.Tensor(ast.literal_eval(englishlist)),englishfiller])
        
        spanishfiller = torch.zeros([self.sp_max - len(spanishlist)])
        output = torch.cat([torch.Tensor(ast.literal_eval(spanishlist)),spanishfiller])
        
        return input.to(torch.int64), output.to(torch.int64)
        
        
        
        

