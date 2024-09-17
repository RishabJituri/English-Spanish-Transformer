from ESPreprocessor import EnglishSpanishPreprocessor

preprocessor = EnglishSpanishPreprocessor("./data/data.csv")
preprocessor.process(
    "./data/processed_data.csv", 
    "./data/english_dict.csv", 
    "./data/spanish_dict.csv",
    "./data/stats.txt"
    )