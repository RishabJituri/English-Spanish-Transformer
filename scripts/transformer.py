import torch.nn as nn
import torch 
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size:int, input_dim:int, heads:int, param_dim:int, final_dim):
        super().__init__()
        self.input_size = input_size
        self.input_dim = input_dim
        self.heads = heads
        self.Qw = torch.rand([heads,input_size.size(-1),param_dim])
        self.Vw = torch.rand([heads,input_size.size(-1),param_dim])
        self.Kw = torch.rand([heads,input_size.size(-1),param_dim])
        self.final_linear = nn.Linear(param_dim*heads,final_dim)
        
    def forward(self,x):
        Q = torch.matmul(x,self.Qw)
        K = torch.matmul(x,self.Kw)
        V = torch.matmul(x,self.Vw)
        score = F.sigmoid(Q@K.transpose(1,2)/K.size(-1))
        scale_dot_attention = torch.matmul(score,V)
        concatenated = scale_dot_attention.reshape(45,-1)
        final = self.final_linear(concatenated)
        return final
            
class Encoder(nn.Module):
    def __init__(self,input_size,embedding_dim,attention_heads,attention_param_dim,attention_linear_dim):
        super().__init__()
        self.attention = MultiHeadAttention(input_size,embedding_dim,attention_heads,attention_param_dim,attention_linear_dim)
        
           
   
     
class Transformer(nn.Module):
    def __init__(self,input_size,embedding_dim):
        self.input_embedding = nn.Embedding(num_embeddings=input_size,embedding_dim=1)
        self.encoder = Encoder()
        self.decoder = Decoder()
        

        
        
    