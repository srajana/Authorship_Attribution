from __future__ import division
from svmutil import *
import math
import os
import glob
import re
import numpy as np
import itertools
import scipy.spatial.distance
from collections import Counter
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.stem.porter import *
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import *

def get_all_files(directory):
	files = []
	for name in os.listdir(directory):
		if os.path.isfile(os.path.join(directory, name)):
			files.append(os.path.join(directory, name))
	return files

def create_excerpt_label_mapping(filename):
	#mapping = {}
	X = []
	Y = []
	f = open(filename, "r") 
	for line in f.readlines():
		excerpt_label = line.split("\t")
		#mapping[excerpt_label[0]] = excerpt_label[1].strip("\n")
		if len(excerpt_label) > 1:
			X.append(excerpt_label[0])
			Y.append(excerpt_label[1])
		else:
			X.append(excerpt_label[0])
			Y.append(1)
	f.close()
	return X, Y
		
def create_vocab(paragraphs):
	sentences = []
	words = []
	stemmer = PorterStemmer()
	for p in paragraphs:
		s = sent_tokenize(p.decode('utf-8'))
		sentences.extend(s)
	for s in sentences:
		w = word_tokenize(s)
		words.extend(w)
	for i in range(len(words)):
		#words[i] = words[i].encode('utf-8').lower()
		words[i] = words[i].lower()
		words[i] = stemmer.stem(words[i])
		words[i] = words[i].encode('utf-8')
	freq_dict = Counter(words)
	top_words = sorted(freq_dict, key=freq_dict.__getitem__, reverse = True)
	'''top_5000 = top_words[1:5000]
	return top_5000'''
	top_1500 = top_words[1:1500]
	return top_1500
	
def get_words(paragraph):
	sentences = []
	words = []
	stemmer = PorterStemmer()
	s = sent_tokenize(paragraph.decode('utf-8'))
	sentences.extend(s)
	for s in sentences:
		w = word_tokenize(s)
		words.extend(w)
	for i in range(len(words)):
		#words[i] = words[i].encode('utf-8').lower()
		words[i] = words[i].lower()
		words[i] = stemmer.stem(words[i])
		words[i] = words[i].encode('utf-8')
	return words
	
def create_feature_space(wordlist):
	feature_space = {}
	for i in range(0,len(wordlist)):	
		feature_space[wordlist[i]] = i+1
	return feature_space	
	
def train_test_model(X_train, Y_train, X_test, Y_test):
	clf = GaussianNB()
	clf.fit(X_train, Y_train)
	labels = clf.predict(X_test)
	accuracy = clf.score(X_test, Y_test)
	return labels, accuracy
	
if __name__ == "__main__":
	
	X_train, Y_train = create_excerpt_label_mapping("/home1/c/cis530/project/data/project_articles_train")
	X_test, Y_test = create_excerpt_label_mapping("/home1/c/cis530/project/data/project_articles_test")
	
	# RUN STRATIFIED K-FOLD CROSS VALIDATION HERE
	X = np.array(X_train)
	Y = np.array(Y_train)
	accuracies = []
	kf = StratifiedKFold(Y, n_folds=10)
	for train_index, test_index in kf:
		train_X, train_Y = X[train_index], Y[train_index]
		test_X, test_Y = X[test_index], Y[test_index] 
		vocab = create_vocab(train_X)
		labels, accuracy= train_test_model(train_X, train_Y, test_X, test_Y)
		accuracies.append(float(accuracy))
	print accuracies
	acc = sum(accuracies)/len(accuracies)
	print acc
	
	'''
	n = 3*len(mapping) // 4   # taking 75% for train and 25% for test currently      
	i = mapping.iteritems()      
	train_mapping = dict(itertools.islice(i, n))   
	test_mapping = dict(i)
	
	train_data = train_mapping.keys()
	vocab = create_vocab(train_data)
	convert_to_svm(train_mapping, 'train.svm', vocab)
	convert_to_svm(test_mapping, 'test.svm', vocab)
	labels, accuracy, values = train_test_model('train.svm', 'test.svm')'''