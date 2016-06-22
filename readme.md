conv.py file performs the training and saves the model parameters in file model.ckpt 
readtraining.py contains functions to preprocess the data and to create word2vec dictionary to be used in the training and in the NER.py script.
NER.py file loads the model saved in file model.ckpt and when executed asks the user to type a query like in the description file

Type a query (type "exit" to exit):
news about Obama

news    B-NEWSTYPE
about   O
Obama   B-KEYWORDS

Type a query (type "exit" to exit):
