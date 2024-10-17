import torch.nn as nn
import torch 
import torch.nn.functional as F
import math
from ESDataset import EnglishSpanishDataset

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
    def __init__(self, K_size,Q_size,V_size, heads:int, param_dim:int,bias=True,mask=False):
        super().__init__()
        self.heads = heads
        self.mask = mask
        self.Qw = torch.rand([heads,Q_size[-1],param_dim])
        self.Kw = torch.rand([heads,K_size[-1],param_dim])
        self.Vw = torch.rand([heads,V_size[-1],V_size[-1]//heads])
        self.final_linear = nn.Linear(V_size[-1],V_size[-1],bias)
        
    def forward(self,tensor_Q,tensor_K,tensor_V):
        Q = torch.matmul(tensor_Q,self.Qw)
        K = torch.matmul(tensor_K,self.Kw)
        V = torch.matmul(tensor_V,self.Vw)
        score = F.sigmoid(Q@K.transpose(1,2)/math.sqrt(K.size(-1)))
        # if self.mask: 
        #     score.masked_fill()
        scale_dot_attention = torch.matmul(score,V)
        print(score.size())
        print(V.size())
        print(scale_dot_attention.size())
        concatenated = scale_dot_attention.reshape(scale_dot_attention.size(1),-1)
        final = self.final_linear(concatenated)
        return final
            
class Encoder(nn.Module):
    def __init__(self,input_size,attention_heads,attention_param_dim):
        super().__init__()
        self.attention = MultiHeadAttention(input_size,input_size,input_size,attention_heads,attention_param_dim)
        self.pwff = PositionWiseFFN(input_size)
    def forward(self,x):
        attention_output = self.attention(x,x,x)
        add = x+attention_output
        LayerNorm = F.layer_norm(add,add.size())
        feed_forward_pass = self.pwff(LayerNorm)
        add = LayerNorm + feed_forward_pass
        return(F.layer_norm(add,add.size()))

class Decoder(nn.Module):
    def __init__(self,input_size,target_size, att1_heads, att2_heads,att1_param, att2_param):
        super().__init__()
        self.masked_attention = MultiHeadAttention(target_size,target_size,target_size,att1_heads,att1_param,True,True)
        self.second_attention = MultiHeadAttention(input_size,target_size,input_size,att2_heads,att2_param)
        self.pwff = PositionWiseFFN(target_size)
    
    def forward(self,input,target):
        masked_attention = self.masked_attention(target,target,target)
        add = masked_attention + target
        LayerNorm = F.layer_norm(add,add.size())
        reg_attention = self.second_attention(LayerNorm,input,input)
        add = reg_attention + LayerNorm
        LayerNorm = F.layer_norm(add,add.size())
        feed_forward_pass = self.pwff(LayerNorm)
        add = feed_forward_pass + LayerNorm
        LayerNorm = F.layer_norm(add,add.size())
        return LayerNorm
        
        
class Transformer(nn.Module):
    def __init__(self,input_size,target_size, embedding_dim):
        super().__init__()
        self.input_embedding = nn.Embedding(num_embeddings=input_size,embedding_dim=embedding_dim)
        self.target_embedding = nn.Embedding(num_embeddings=target_size,embedding_dim=embedding_dim)
        self.encoder = Encoder()
        self.decoder = Decoder()



statsfile = "../data/stats.txt"
datafile = "../data/processed_data.csv"
Dataset = EnglishSpanishDataset(statsfile, datafile)

input,output = Dataset[1]

embedding_type_beat = nn.Embedding(num_embeddings=input.size(-1),embedding_dim=512)
x = embedding_type_beat(input)
print(x.size())
embedding_type_beat2 = nn.Embedding(num_embeddings=output.size(-1),embedding_dim=512)
y = embedding_type_beat2(output)

encoder = Encoder(x.size(),8,256)
encoder(x)
decoder = Decoder(x.size(),y.size(),8,8,256,256)

print(decoder(x,y).size())
        
        
    