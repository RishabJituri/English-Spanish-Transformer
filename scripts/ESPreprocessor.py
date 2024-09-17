import csv
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd

class EnglishSpanishPreprocessor():
    def __init__(self, csvpath):
        self.dataset = pd.read_csv(csvpath)
        self.english_dict = {"<END>" : 0}
        self.spanish_dict = {"<END>" : 0}
        self.max_english_sentance = 0
        self.max_spanish_sentance = 0
    
    def __convert__(self,sent, language):
        tokens = []
        dictionary = None
        if language == "english": dictionary = self.english_dict
        else: dictionary = self.spanish_dict
        for word in sent:
            token = None
            if word in dictionary: 
                token = dictionary[word]
            else: 
                dictionary[word] = len(dictionary)
                token = dictionary[word]
            tokens.append(token)
        if language == "english":
            if len(tokens)>self.max_english_sentance: self.max_english_sentance = len(tokens)
        else: 
            if len(tokens)>self.max_spanish_sentance: self.max_spanish_sentance = len(tokens)
        
        return tokens
                    
    def __delete_pd__(self):
        del self.dataset
                  
    
    def __write_to_file__(self,writer,row):
        english= word_tokenize(row["english"],"english")
        spanish=word_tokenize(row["spanish"],"spanish")
        
        english_numerical = self.__convert__(english,"english")
        spanish_numerical = self.__convert__(spanish,"spanish")
        
        writer.writerow({"english" : english_numerical, "spanish" : spanish_numerical})
    
    def process(self,dataset_filename, english_dictionary_filename, spanish_dictionary_filename,stats_filename):
        
        with open(dataset_filename,mode='w') as csvfile:
            writer = csv.DictWriter(csvfile,fieldnames=["english","spanish"])
            writer.writeheader()
            for _,row in self.dataset.iterrows():
                self.__write_to_file__(writer,row)
        
        
        with open(english_dictionary_filename,mode="w") as csvfile:
            writer = csv.writer(csvfile)
            writer = csv.DictWriter(csvfile,fieldnames=["word","token"])
            writer.writeheader()
            for key in (self.english_dict):
                value = (self.english_dict[key])
                writer.writerow({
                    "word" : key,
                    "token": value
                    })
        
        with open(spanish_dictionary_filename,mode="w") as csvfile:
            writer = csv.writer(csvfile)
            writer = csv.DictWriter(csvfile,fieldnames=["word","token"])
            writer.writeheader()
            for key in (self.spanish_dict):
                value = (self.spanish_dict[key])
                
                writer.writerow({
                    "word" : key,
                    "token": value
                    })
        f= open(stats_filename, mode="w")
        f.write(str(self.max_english_sentance))
        f.write("\n")
        f.write(str(self.max_spanish_sentance))
        
            