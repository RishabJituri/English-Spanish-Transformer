import torch.nn as nn
import torch 
import torch.nn.functional as F



class Encoder(nn.Module):
    def __init__(self,input_size,embedding_dim):
        super().__init__()
        
        

        

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size,input_dim):
        super().__init__()
        self.input_size
        self.Qw = nn.Linear()
        self.Vw = nn.Linear()
        self.Kw = nn.Linear()
        self.final_linear = nn.Linear()
        
    def forward(self,x):
        concatenated = None 
        for i in range(self.input_size):
            Q = self.Qw(x[i])
            V = self.Vw(x[i])
            K = self.Kw(x[i])
            ScaleDotProduct = F.softmax(Q@K/K.size(-1))*V
            if concatenated == None:
                concatenated = ScaleDotProduct.unsqueeze(0)
            else:
                concatenated = torch.concatenate([concatenated,ScaleDotProduct.unsqueeze(0)],dim = 0)
            
        
class Transformer(nn.Module):
    def __init__(self,input_size,embedding_dim):
        self.input_embedding = nn.Embedding(num_embeddings=input_size,embedding_dim=1)
        self.encoder = Encoder()
        self.decoder = Decoder()
        

        
        
    