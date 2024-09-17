import torch
from torch.utils.data import Dataset
import pandas as pd

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
        print()
        temp = torch.Tensor(englishlist)
        input = torch.cat([temp,englishfiller])
        
        spanishfiller = torch.zeros([self.sp_max - len(spanishlist)])
        output = torch.cat[torch.Tensor(spanishlist),spanishfiller]
        
        return input, output
        
        
        
        

