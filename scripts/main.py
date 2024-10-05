from ESPreprocessor import EnglishSpanishPreprocessor
from ESDataset import EnglishSpanishDataset
import argparse
import sys
import torch.nn as nn


statsfile = "./data/stats.txt"
datafile = "./data/processed_data.csv"
Dataset = EnglishSpanishDataset(statsfile, datafile)

input,_ = Dataset[0]
embedding_type_beat = nn.Embedding(num_embeddings=input.size(-1),embedding_dim=512)
print(embedding_type_beat(input))




