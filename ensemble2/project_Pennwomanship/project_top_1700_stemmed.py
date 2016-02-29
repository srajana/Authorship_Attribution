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
	top_5000 = top_words[1:1701]
	return top_5000
	'''top_1500 = top_words[1:1500]
	return top_1500'''
	
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
	
def convert_to_svm(excerpts, labels, outfile, vocab):
	fs = create_feature_space(vocab)
	f = open(outfile, "w")
	for i,entry in enumerate(excerpts):
		feature_vector = []
		words = get_words(entry)
		new_words = []
		for w in words:
			if w in vocab:
				new_words.append(w)
		freq_dict = Counter(new_words)
		for w in new_words:
			pair = str(fs[w]) + ":" + str(freq_dict[w])
			feature_vector.append(pair)
		f.write(str(int(labels[i]) + 1) + "\t" + ' '.join(feature_vector) + "\n")
	f.close()
	
def train_test_model(train_datafile, test_datafile):
	y1,x1 = svm_read_problem(train_datafile)
	m = svm_train(y1, x1, '-t 0 -e .01 -m 1000 -h 0')
	y2,x2 = svm_read_problem(test_datafile)
	label, accuracy, values = svm_predict(y2, x2, m)
	return label, accuracy, values
	
def predict():
	X_train, Y_train = create_excerpt_label_mapping("/home1/c/cis530/project/data/project_articles_train")
	X_test, Y_test = create_excerpt_label_mapping("/home1/c/cis530/project/data/project_articles_test")
	vocab = create_vocab(X_train)
	convert_to_svm(X_train, Y_train, 'train.svm', vocab)
	convert_to_svm(X_test, Y_test, 'test.svm', vocab)
	labels, accuracy, values = train_test_model('train.svm', 'test.svm')
	for i in range(len(labels)):
		labels[i] = int(labels[i])-1
	return labels
	
if __name__ == "__main__":
	
	labels = predict()
	'''
	X_train, Y_train = create_excerpt_label_mapping("/home1/c/cis530/project/data/project_articles_train")
	X_test, Y_test = create_excerpt_label_mapping("/home1/c/cis530/project/data/project_articles_test")
	print len(X_train)
	print len(Y_train)
	print len(X_test)
	print len(Y_test)
	print sum(Y_test)/len(Y_test)
	vocab = create_vocab(X_train)
	convert_to_svm(X_train, Y_train, 'train.svm', vocab)
	convert_to_svm(X_test, Y_test, 'test.svm', vocab)
	labels, accuracy, values = train_test_model('train.svm', 'test.svm')
	f = open("test.txt","w")
	for l in labels:
		f.write(str(int(l-1))+"\n")
	f.close()
	
	
	# RUN K-FOLD CROSS VALIDATION HERE
	X = np.array(X_train)
	Y = np.array(Y_train)
	accuracies = []
	kf = KFold(len(X_train), n_folds=10)
	for train_index, test_index in kf:
		train_X, train_Y = X[train_index], Y[train_index]
		test_X, test_Y = X[test_index], Y[test_index] 
		vocab = create_vocab(train_X)
		convert_to_svm(train_X, train_Y, 'train.svm', vocab)
		convert_to_svm(test_X, test_Y, 'test.svm', vocab)
		labels, accuracy, values = train_test_model('train.svm', 'test.svm')
		accuracies.append(float(accuracy[0]))
	print accuracies
	acc = sum(accuracies)/len(accuracies)
	print acc
	
	# RUN STRATIFIED K-FOLD CROSS VALIDATION HERE
	X = np.array(X_train)
	Y = np.array(Y_train)
	accuracies = []
	kf = StratifiedKFold(Y, n_folds=10)
	for train_index, test_index in kf:
		train_X, train_Y = X[train_index], Y[train_index]
		test_X, test_Y = X[test_index], Y[test_index] 
		vocab = create_vocab(train_X)
		convert_to_svm(train_X, train_Y, 'train.svm', vocab)
		convert_to_svm(test_X, test_Y, 'test.svm', vocab)
		labels, accuracy, values = train_test_model('train.svm', 'test.svm')
		accuracies.append(float(accuracy[0]))
	print accuracies
	acc = sum(accuracies)/len(accuracies)
	print acc'''
