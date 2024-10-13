import torch.nn as nn
import torch 
import torch.nn.functional as F
import math

class PositionWiseFFN(nn.Module):
    def __init__(self,input_size,bias=True):
        super().__init__()
        self.first_layer = nn.Linear(input_size[-1],input_size[-1]*2,bias)
        self.second_layer = nn.Linear(input_size[-1]*2,input_size[-1],bias)
    def forward(self,x):
        x = self.first_layer(x)
        x = F.relu(x)
        x = self.second_layer(x)
        return x
        

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size:int, input_dim:int, heads:int, param_dim:int,bias=True,mask=False):
        super().__init__()
        self.input_size = input_size
        self.input_dim = input_dim
        self.heads = heads
        self.Qw = torch.rand([heads,input_size.size(-1),param_dim])
        self.Kw = torch.rand([heads,input_size.size(-1),param_dim])
        self.Vw = torch.rand([heads,input_size.size(-1),input_dim//heads])
        self.final_linear = nn.Linear(param_dim*heads,param_dim*heads,bias)
        
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
    def __init__(self,input_size,attention_heads,attention_param_dim,attention_linear_dim):
        super().__init__()
        self.attention = MultiHeadAttention(input_size[0],input_size[-1],attention_heads,attention_param_dim,attention_linear_dim)
        self.pwff = PositionWiseFFN(input_size)
    def forward(self,x):
        attention_output = self.attention(x)
        add = x+attention_output
        LayerNorm = F.layer_norm(add,add.size())
        feed_forward_pass = self.pwff(LayerNorm)
        add = LayerNorm + feed_forward_pass
        return(F.layer_norm(add,add.size()))

class Decoder(nn.Module):
    def __init__(self,input_size,target_size)
        
        
           
   
     
class Transformer(nn.Module):
    def __init__(self,input_size,embedding_dim):
        self.input_embedding = nn.Embedding(num_embeddings=input_size,embedding_dim=1)
        self.encoder = Encoder()
        self.decoder = Decoder()
        

        
        
    