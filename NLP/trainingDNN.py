# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 15:18:11 2018

@author: vaisakhs
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

#LOAD DATA
newsgroups_train = fetch_20newsgroups(subset='train')
print(list(newsgroups_train.target_names))

#VECTORIZE
categories = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train',
                                      categories=categories)
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)

# Save Tf Vectorizer Model
pickle.dump(vectorizer,open(r"C:\Users\vaisakhs\Desktop\GITHUB\adversarial_Machine_Learning\NLP\model\tfidf.pickle","wb"),protocol=4) 

#FEATURE SELECTION
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
lsvc = LinearSVC(C=0.7, penalty="l1", dual=False).fit(vectors, newsgroups_train.target)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(vectors)
print(X_new.shape)
pos=model.get_support(indices=True)
pickle.dump(pos,open(r"C:\Users\vaisakhs\Desktop\GITHUB\adversarial_Machine_Learning\NLP\model\positions.pickle","wb"),protocol=4)               
            
                
#FIT CLASSIFIER
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
newsgroups_test = fetch_20newsgroups(subset='test',
                                      categories=categories)
vectors_test = vectorizer.transform(newsgroups_test.data)
clf = MultinomialNB(alpha=.01)
clf.fit(X_new, newsgroups_train.target)
pred = clf.predict(vectors_test[:,pos])
metrics.f1_score(newsgroups_test.target, pred, average='macro')
