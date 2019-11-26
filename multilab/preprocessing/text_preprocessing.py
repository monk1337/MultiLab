import pickle as pk
import sys
import warnings
import operator
import pandas as pd
import string
from collections import Counter
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
import random
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import os

class Text_preprocessing(object):


    def __init__(self):

        self.stop_words     = set(stopwords.words('english'))
        self.re_stop_words  = re.compile(r"\b(" + "|".join(self.stop_words) + ")\\W", re.I)
        self.stemmer        = SnowballStemmer("english")
    
    #return sentences and labels 
    def initial_preprocess(self, df_, stop_w = False, stem_ = False, chunk_value = False):

        self.df              = df_
        self.chunk_value     = chunk_value
        self.stem_           = stem_
        self.stop_w          = stop_w
        
        # todo : progress bar for pandas internal process
        # not helpful https://stackoverflow.com/questions/18603270/progress-indicator-during-pandas-operations

        print('lower_case done')
        self.df['text']      = self.df['text'].str.lower()
        self.df['text']      = self.df['text'].apply(self.remove_pun)
        tqdm.pandas()

        print('punctuation removed')

        self.df['text']      = self.df['text'].apply(self.keep_alpha)
        self.df['text']      = self.df['text'].apply(self.clean_html)
        tqdm.pandas()

        print('text cleaning done')
        if self.stop_w:
            self.df['text'] = self.df['text'].apply(self.removeStopWords)
        elif self.stem_:
            self.df['text'] = self.df['text'].apply(self.stemming)
        elif self.chunk_value:
            self.df['text'] = self.df['text'].apply(self.chunk, value = self.chunk_value)
        
        return self.df

    #lowercase
    def lower_case(self, df, column_name = 'text'):
        df[column_name] = df[column_name].str.lower()
        return df

    #slice dataset 
    def dataset_slice(self, df, ratio = False, no_of_samples = False):

        if ratio:
            sample_df = df.sample(frac = float(ratio))
        elif no_of_samples:
            sample_df = df.sample(n = int(no_of_samples))
        return sample_df.reset_index(drop=True)

    def split_dataframe():
        pass
    
    #remove html 
    def clean_html(self,sentence):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, ' ', str(sentence))
        return cleantext
    
    #remove punctuation
    def remove_pun(self,sentence):
        return " ".join(sentence.translate(str.maketrans('', '', string.punctuation)).split())
    

    # only keep alpha
    def keep_alpha(self,sentence):
        alpha_sent = ""
        for word in sentence.split():
            alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
            alpha_sent += alpha_word
            alpha_sent += " "
        alpha_sent = alpha_sent.strip()
        return alpha_sent

    # remove stop words
    def removeStopWords(self, sentence):
        return self.re_stop_words.sub(" ", sentence)
    
    # stem over sentences
    def stemming(self,sentence):
        stemSentence = ""
        for word in sentence.split():
            stem = self.stemmer.stem(word)
            stemSentence += stem
            stemSentence += " "
        stemSentence = stemSentence.strip()
        return stemSentence
    
    
    # get_sentence length
    def get_sentence_length(self,df_):
        
        sentences = list(df_['text'])

    
        lenths = []
        
        for sentence in tqdm(sentences):
            if isinstance(sentence,list):
                lenths.append(len(sentence))
            else:
                lenths.append(len(sentence.split()))
        return "max {} min {} average {}".format(np.max(np.array(lenths)), 
                                                np.min(np.array(lenths)), 
                                                np.average(np.array(lenths)))
    
    def chunk(self, sentence, value):
        return " ".join(sentence.split()[:value])
    


    def keep_labels(self, df, keep_ratio = False, freq_Value = False):
    
        text_col = df['text']
        df_ = df.drop('text', 1)
        
        get_frequency = {}
        
        for column in df_.columns:
            get_frequency[column]= (df_[column]==1).sum()
        sorted_long   = sorted(get_frequency.items(), key=operator.itemgetter(1),reverse=True)
        
        if freq_Value:
            sorted_long = sorted([col for col in sorted_long if int(col[1])>= freq_Value])
            
        if keep_ratio:
            keep_ratio   = int(len(sorted_long)* keep_ratio)
            sorted_long  = sorted(sorted_long[: keep_ratio])
        
        keep_columns = [col[0] for col in sorted_long]
        
        
        df_          = df_[sorted(keep_columns)]
        df_['text']  = text_col
        
        # remove all rows where all labels are zeros
        df_          = df_[(df_.loc[:, df_.columns != 'text'].T != 0).any()]
        df_          = df_.reset_index(drop=True)
        
        # return dataframe and labels frequency
        return df_, sorted_long
    
    # glove embedding
    def loadGloveModel(self, gloveFile):
        print("Loading Glove Model")
        f = open(gloveFile,'r')
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print("Done.",len(model)," words loaded!")
        return model
    
    

    # first get the vocab frequency by passing sentences only 
    # then give the keep ratio, freq value to how many words you want in vocab

    def vocab_freq(self, 
               df_data, 
               keep_ratio = False, 
               freq_Value = False, 
               custom_value = False):
               
        
        sentences = list(df_data['text'])
        vocab_frequency = []
        for sentence in tqdm(sentences):
            if isinstance(sentence,list):
                vocab_frequency.extend(sentence)
            else:
                vocab_frequency.extend(sentence.split())
        freq = Counter(vocab_frequency)
        
        sorted_long   = sorted(freq.items(), key=operator.itemgetter(1),reverse=True)
        
        if freq_Value:
            sorted_long = sorted([col for col in sorted_long if int(col[1])>= freq_Value])
            
        if keep_ratio:
            keep_ratio   = int(len(sorted_long)* keep_ratio)
            sorted_long  = sorted(sorted_long[:keep_ratio])
            
        if custom_value:
            sorted_long = sorted(sorted_long[:custom_value])
            
        freq_num        = set([k[1] for k in sorted_long])
        
        word_to_int = [(n[0],s) for s,n in enumerate(sorted_long,2)]
        word_to_int.extend([('unk',1),('pad',0)])
        int_to_word = [(n,m) for m,n in word_to_int]
            
        return sorted_long, freq_num,word_to_int,int_to_word


     # get the encoded dataset
    def encoder(self, df_, vocab_dict):

        sentences = list(df_['text'])
        labels    = np.array(df_.drop('text', 1))

        vocab_dict = dict(vocab_dict)
        all_sentences = []
        
        for sentence in tqdm(sentences):
            token = nltk.word_tokenize(sentence)
            encoded_token = []
            for k in token:
                if k in vocab_dict:
                    
                    encoded_token.append(vocab_dict[k])
                else:
                    encoded_token.append(vocab_dict['unk'])
            all_sentences.append(encoded_token)
        
        return all_sentences, labels, vocab_dict
    
    # tf-ifd vectors
    def tf_idf(self, df_frame):
        
        sentences = df_frame['text']
        labels    = np.array(df_frame.drop('text', 1))
        
        
        vectorizer = TfidfVectorizer(strip_accents='unicode', 
                                    analyzer='word', 
                                    ngram_range=(1,3), 
                                    norm='l2')
        vectorizer.fit(sentences)
        x_train = vectorizer.transform(sentences)
        
        return x_train, labels

    def vocab_embedding(self, vocab, path):

        """
        Does neat stuff!
        Parameters:
        foo - vocab should be in tuple format where ( word, frequency_of_word ) ex : [(hello,29),(world,32)]
        path - path of glove embedding
        """

        load_embeddings = loadGloveModel(path)
        vocab   = sorted(dict(vocab).items(), key=operator.itemgetter(1))
            
        encoded_vocab    = []
        not_in_embedding = []

        vocas = [token[0].lower() for token in vocab]
        
        for token in tqdm(vocas):
            if token in load_embeddings:
                encoded_vocab.append(load_embeddings[token])
            else:
                not_in_embedding.append(token)
                encoded_vocab.append(load_embeddings['unk'])
        
        return np.array(encoded_vocab), not_in_embedding


     # if datalabels are like [ [ 'hello','world'], ['hello','hello','hello']] 
     # convert them into onehot   
    def labels_to_dataframe(self, sentences,labels):

        mlb              = MultiLabelBinarizer()
        labels_on        = mlb.fit_transform(labels)
        pd_data          = pd.DataFrame(labels_on)
        pd_data.columns  = mlb.classes_
        pd_data['text']  = sentences
        return pd_data
    

    

    

        


    

