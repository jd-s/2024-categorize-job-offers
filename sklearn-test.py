import sqlite3
import pandas as pd
import numpy as np
#from ipydatagrid import DataGrid
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import spacy
import re
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np#for text pre-processing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')#for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score# bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer#for word embedding
import gensim
from gensim.models import Word2Vec
import seaborn as sns


conn = sqlite3.connect('/home/jens/tmp/database2.db')
qu = """SELECT * FROM Tweets where Tweets.job_id LIKE'%B 4%' and 1"""
df_train = pd.read_sql(qu, conn)   
df_train.createdAt = pd.to_datetime(df_train.createdAt)
for index, row in df_train.iterrows():
    df_train.at[index,'jahr'] = row.createdAt.year
    
from sklearn.utils import resample

minority_class = df_train[df_train['category'] <= 4]
majority_class = df_train[df_train['category'] == 5]

# Downsample the majority class
majority_downsampled = resample(majority_class, replace=False, n_samples=300, random_state=42)

# Combine the downsampled majority class with the minority class
df_train = pd.concat([minority_class, majority_downsampled ])

# WORD-COUNT
df_train['word_count'] = df_train['tweet_txt'].apply(lambda x: len(str(x).split()))
print(df_train[df_train['category']==0]['word_count'].mean()) #Disaster tweets
print(df_train[df_train['category']==1]['word_count'].mean()) #Non-Disaster tweets
print(df_train[df_train['category']==3]['word_count'].mean()) #Non-Disaster tweets
print(df_train[df_train['category']==4]['word_count'].mean()) #Non-Disaster tweets
print(df_train[df_train['category']==5]['word_count'].mean()) #Non-Disaster tweets

#convert to lowercase, strip and remove punctuations
def preprocess(text):
    text = text.lower() 
    text=text.strip()  
    text=re.compile('<.*?>').sub('', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text) 
    return text
 
# STOPWORD REMOVAL
def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)#LEMMATIZATION
# Initialize the lemmatizer
wl = WordNetLemmatizer()
 
# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN# Tokenize the sentence
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)
    
nlp = spacy.load("de_core_news_sm", disable=['parser', 'tagger', 'ner'])
stops = stopwords.words("german")

def normalize(comment, lowercase, remove_stopwords):
    if lowercase:
        comment = comment.lower()
    comment = nlp(comment)
    lemmatized = list()
    for word in comment:
        lemma = word.lemma_.strip()
        if lemma:
            if not remove_stopwords or (remove_stopwords and lemma not in stops):
                lemmatized.append(lemma)
    return " ".join(lemmatized)
    
df_train['clean_text'] = df_train['tweet_txt'].apply(normalize, lowercase=True, remove_stopwords=True)
#SPLITTING THE TRAINING DATASET INTO TRAIN AND TEST
X_train, X_test, y_train, y_test = train_test_split(df_train["clean_text"],df_train["category"],test_size=0.2,shuffle=True)#Word2Vec
# Word2Vec runs on tokenized sentences
X_train_tok= [nltk.word_tokenize(i, 'german') for i in X_train]  
X_test_tok= [nltk.word_tokenize(i, 'german') for i in X_test]

#Tf-Idf
tfidf_vectorizer = TfidfVectorizer(use_idf=True)
X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train) 
X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test)#building Word2Vec model
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))
    def fit(self, X, y):
        return self
    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
            ])

df_train['clean_text_tok']=[nltk.word_tokenize(i, 'german') for i in df_train['clean_text']]
model = Word2Vec(df_train['clean_text_tok'],min_count=1)     
w2v = dict(zip(model.wv.index_to_key, model.wv.vectors)) 
modelw = MeanEmbeddingVectorizer(w2v)# converting text to numerical data using Word2Vec
X_train_vectors_w2v = modelw.transform(X_train_tok)
X_val_vectors_w2v = modelw.transform(X_test_tok)

#FITTING THE CLASSIFICATION MODEL using Logistic Regression(tf-idf)
lr_tfidf=LogisticRegression(solver = 'liblinear', C=10, penalty = 'l2')
lr_tfidf.fit(X_train_vectors_tfidf, y_train)  #Predict y value for test dataset

y_predict = lr_tfidf.predict(X_test_vectors_tfidf)
y_prob = lr_tfidf.predict_proba(X_test_vectors_tfidf)[:,1]

print(classification_report(y_test,y_predict))
print(f1_score(y_test,y_predict, average='weighted'))
print('Confusion Matrix:',confusion_matrix(y_test, y_predict))

roc_auc=roc_auc_score(y_train, lr_tfidf.predict_proba(X_train_vectors_tfidf),  multi_class='ovr')
#fpr, tpr, thresholds = roc_curve(y_test, y_prob)
#roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)

roc_auc=roc_auc_score(y_train, lr_tfidf.predict_proba(X_train_vectors_tfidf),  multi_class='ovr')
#fpr, tpr, thresholds = roc_curve(y_test, y_prob)
#roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)

#FITTING THE CLASSIFICATION MODEL using Logistic Regression (W2v)

lr_w2v=LogisticRegression(solver = 'liblinear', C=10, penalty = 'l2')
lr_w2v.fit(X_train_vectors_w2v, y_train)  #model#Predict y value for test dataset

y_predict = lr_w2v.predict(X_val_vectors_w2v)
y_prob = lr_w2v.predict_proba(X_val_vectors_w2v)[:,1]
print(classification_report(y_test,y_predict))
print(f1_score(y_test,y_predict, average='weighted'))
print('Confusion Matrix:',confusion_matrix(y_test, y_predict))

#FITTING THE CLASSIFICATION MODEL using Naive Bayes(tf-idf)
nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_vectors_tfidf, y_train)  #Predict y value for test dataset
y_predict = nb_tfidf.predict(X_test_vectors_tfidf)
y_prob = nb_tfidf.predict_proba(X_test_vectors_tfidf)[:,1]
print(classification_report(y_test,y_predict))
print(f1_score(y_test,y_predict, average='weighted'))

print('Confusion Matrix:',confusion_matrix(y_test, y_predict))


roc_auc=roc_auc_score(y_train, nb_tfidf.predict_proba(X_train_vectors_tfidf),  multi_class='ovr')
#fpr, tpr, thresholds = roc_curve(y_test, y_prob)
#roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)

# creating a RF classifier
clf = RandomForestClassifier(n_estimators = 100)  
clf.fit(X_train_vectors_tfidf, y_train)  #Predict y value for test dataset
y_predict = clf.predict(X_test_vectors_tfidf)
y_prob = clf.predict_proba(X_test_vectors_tfidf)[:,1]
print(classification_report(y_test,y_predict))

