import torch.nn as nn
import torch 
import torch.nn.functional as F
import math

class PositionWiseFFN(nn.Module):
    def __init__(self,):
        super().__init__()
        

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size:int, input_dim:int, heads:int, param_dim:int):
        super().__init__()
        self.input_size = input_size
        self.input_dim = input_dim
        self.heads = heads
        self.Qw = torch.rand([heads,input_size.size(-1),param_dim])
        self.Kw = torch.rand([heads,input_size.size(-1),param_dim])
        self.Vw = torch.rand([heads,input_size.size(-1),input_dim//heads])
        self.final_linear = nn.Linear(param_dim*heads,param_dim*heads)
        
    def forward(self,x):
        Q = torch.matmul(x,self.Qw)
        K = torch.matmul(x,self.Kw)
        V = torch.matmul(x,self.Vw)
        score = F.sigmoid(Q@K.transpose(1,2)/math.sqrt(K.size(-1)))
        scale_dot_attention = torch.matmul(score,V)
        concatenated = scale_dot_attention.reshape(45,-1)
        final = self.final_linear(concatenated)
        return final
            
class Encoder(nn.Module):
    def __init__(self,input_size,embedding_dim,attention_heads,attention_param_dim,attention_linear_dim):
        super().__init__()
        self.attention = MultiHeadAttention(input_size,embedding_dim,attention_heads,attention_param_dim,attention_linear_dim)
        self.layerNorm = nn.LayerNorm()
    def forward(self,x):
        attention_output = self.attention(x)
        add_plus_layerNorm = self.layerNorm(x+attention_output)
        feed_forward_pass = 
        
        
           
   
     
class Transformer(nn.Module):
    def __init__(self,input_size,embedding_dim):
        self.input_embedding = nn.Embedding(num_embeddings=input_size,embedding_dim=1)
        self.encoder = Encoder()
        self.decoder = Decoder()
        

        
        
    