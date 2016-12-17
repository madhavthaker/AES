from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.stats import spearmanr
import nltk.data
import re
from nltk.corpus import stopwords
import tensorflow as tf
import logging
from gensim.models import word2vec
from sklearn.metrics import cohen_kappa_score
import timeit
from sklearn.model_selection import train_test_split

xl_workbook = pd.ExcelFile('training_set_rel3.xlsx')
df_all = xl_workbook.parse("training_set")
df_all = df_all[df_all['domain1_score']<61]
df_all = df_all.dropna(axis = 1)
df_all = df_all.drop('rater1_domain1', 1)
df_all = df_all.drop('rater2_domain1', 1)
df_all = df_all.drop('essay_id', 1)

X_train, X_test, y_train, y_test = train_test_split(df_all['essay'], df_all['domain1_score'], test_size=0.10)
y_train = np.reshape(y_train, (len(y_train), 1))
y_test = np.reshape(y_test, (len(y_test), 1))

def essay_to_wordlist(essay_v, remove_stopwords):
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)
    words = essay_v.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def essay_to_sentences(essay_v, remove_stopwords):
    raw_sentences = tokenizer.tokenize(essay_v.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

sentences = []

for essay_v in X_train:    
    sentences += essay_to_sentences(essay_v, remove_stopwords = True)

for essay_v in X_test:
    sentences += essay_to_sentences(essay_v, remove_stopwords = True)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

num_features = 300 
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3

print ("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)

model.init_sims(replace=True)

model_name = "300features_40minwords_10context"
model.save(model_name)

def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])        
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def getAvgFeatureVecs(essays, model, num_features):
    counter = 0.
    essayFeatureVecs = np.zeros((len(essays),num_features),dtype="float32")
    for essay in essays:
        essayFeatureVecs[counter] = makeFeatureVec(essay, model, num_features)
        counter = counter + 1.
    return essayFeatureVecs

print ("Creating average feature vecs for Training Essays")
clean_train_essays = []
for essay_v in X_train:
    clean_train_essays.append( essay_to_wordlist( essay_v, remove_stopwords=True ))
trainDataVecs = getAvgFeatureVecs( clean_train_essays, model, num_features )


clean_test_essays = []
for essay_v in X_test:
    clean_test_essays.append( essay_to_wordlist( essay_v, remove_stopwords=True ))
testDataVecs = getAvgFeatureVecs( clean_test_essays, model, num_features )

