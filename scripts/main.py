from ESPreprocessor import EnglishSpanishPreprocessor
from ESDataset import EnglishSpanishDataset
import argparse
import sys



statsfile = "./data/stats.txt"
datafile = "./data/processed_data.csv"
Dataset = EnglishSpanishDataset(statsfile, datafile)

print(Dataset[0])





