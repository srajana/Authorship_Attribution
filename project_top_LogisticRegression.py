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
from sklearn.linear_model import *

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
			Y.append(int(excerpt_label[1]))
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
		s = sent_tokenize(str(p).decode('utf-8'))
		sentences.extend(s)
	for s in sentences:
		w = word_tokenize(s)
		words.extend(w)
	for i in range(len(words)):
		words[i] = words[i].encode('utf-8').lower()
	freq_dict = Counter(words)
	top_words = sorted(freq_dict, key=freq_dict.__getitem__, reverse = True)
	'''top_5000 = top_words[1:5000]
	return top_5000'''
	top_1500 = top_words[1:1001]
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
		words[i] = words[i].encode('utf-8').lower()
	return words
	
def create_feature_space(wordlist):
	feature_space = {}
	for i in range(0,len(wordlist)):	
		feature_space[wordlist[i]] = i
	return feature_space	

def convert_to_LR(X, vocab):
	rows = []
	fs = create_feature_space(vocab)
	for i,entry in enumerate(X):
		feature_vector = []
		words = get_words(str(entry))
		new_words = []
		for w in words:
			if w in vocab:
				new_words.append(w)
		wc = Counter(new_words)
		for w in vocab:
			feature_vector.append(wc[w])
		rows.append(feature_vector)
	return rows
	
def train_test_model(X_train, Y_train, X_test, Y_test):
	clf = LogisticRegression()
	clf.fit(X_train, Y_train)
	labels = clf.predict(X_test)
	accuracy = clf.score(X_test, Y_test)
	return labels, accuracy
	
if __name__ == "__main__":
	
	X_train, Y_train = create_excerpt_label_mapping("/home1/c/cis530/project/data/project_articles_train")
	X_test, Y_test = create_excerpt_label_mapping("/home1/c/cis530/project/data/project_articles_test")
	vocab = create_vocab(X_train)
	X_train = convert_to_LR(X_train, vocab)
	X_test = convert_to_LR(X_test, vocab)
	labels, accuracy= train_test_model(X_train, Y_train, X_test, Y_test)
	print labels
	print type(labels)
	f = open("lr-labels.txt", "w")
	for i in labels:
		f.write(str(int(i)).decode("utf-8") + "\n")
	f.close()
	print accuracy
	
	# RUN STRATIFIED K-FOLD CROSS VALIDATION HERE
	X = np.array(X_train)
	Y = np.array(Y_train)
	accuracies = []
	kf = StratifiedKFold(Y, n_folds=10)
	for train_index, test_index in kf:
		train_X, train_Y = X[train_index], Y[train_index]
		test_X, test_Y = X[test_index], Y[test_index] 
		vocab = create_vocab(train_X)
		train_X = convert_to_LR(train_X, vocab)
		test_X = convert_to_LR(test_X, vocab)
		#train_X = np.array(train_X)
		#test_X = np.array(test_X)
		#train_X = np.resize(train_X,(len(train_X),1500))
		#test_X = np.resize(test_X,(len(test_X),1500))
		labels, accuracy= train_test_model(train_X, train_Y, test_X, test_Y)
		accuracies.append(float(accuracy))
	print accuracies
	acc = sum(accuracies)/len(accuracies)
	print acc