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
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
import pickle

def get_all_files(directory):
	files = []
	for name in os.listdir(directory):
		if os.path.isfile(os.path.join(directory, name)):
			files.append(os.path.join(directory, name))
	return files

def get_words(paragraph):
	sentences = []
	words = []
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
		feature_space[wordlist[i]] = i+1
	return feature_space	


def standardize(rawexcerpt):
	lowercase = rawexcerpt.lower()
	tokens = word_tokenize(lowercase.decode('utf-8'))
	for s in range(0,len(tokens)):
		tokens[s] = tokens[s].encode('utf-8')
	return tokens

def load_file_excerpts(filepath):
	list_file_excerpts = []
	f = open(filepath, "r")
	for line in f.readlines():
		excerpts = standardize(line)
		list_file_excerpts.append(excerpts)
	f.close()
	return list_file_excerpts

def load_directory_excerpts(dirpath):
	files_in_dir = get_all_files(dirpath)
	list_dir_excerpts = []
	for file in files_in_dir:
		f = open(file, "r")
		for line in f.readlines():
			excerpts = standardize(line)
			list_dir_excerpts.append(excerpts)
		f.close()
	return list_dir_excerpts

def flatten(listoflists):
	flat_list = []
	for i in listoflists:
		flat_list.extend(i)
	return flat_list

def get_tf(sample):
	tf_dict = dict(Counter(sample))
	return tf_dict

def get_idf(corpus):
	n = len(corpus)
	df_dict = {}
	idf_dict = {}
	words = flatten(corpus)
	unique_words = list(set(words))
	for i in unique_words:
		df_dict[i] = 0
		idf_dict[i] = 0
	for i in unique_words:
		for j in corpus:
			if i in j:
				df_dict[i] = df_dict[i]+1
	for i in unique_words:
		idf_dict[i] = math.log(n/df_dict[i])
	return idf_dict

def get_tfidf(tf_dict, idf_dict):
	tf_idf_dict = {}
	for i in tf_dict:
		tf_idf_dict[i] = tf_dict[i] * idf_dict[i]
	return tf_idf_dict

def get_tfidf_weights_topk(tf_dict, idf_dict, k):
	tf_idf_dict = get_tfidf(tf_dict, idf_dict)
	rank_list = []
	rank_list = sorted(tf_idf_dict, key=tf_idf_dict.__getitem__, reverse=True)
	tfidf_weights_topk = []
	for i in range(0,k):
		tfidf_weights_topk.append((rank_list[i],tf_idf_dict[rank_list[i]]))
	return tfidf_weights_topk
		
def get_tfidf_topk(sample, corpus, k):
	tf_dict = get_tf(sample)
	try:
		fileObject = open("idf-pickle1","r")
		idf_dict = pickle.load(fileObject)
		fileObject.close()
	except IOError as e:
		idf_dict = get_idf(corpus)
		fileObject = open("idf-pickle1","wb")
		pickle.dump(idf_dict, fileObject)
		fileObject.close()
	tfidf_topk = get_tfidf_weights_topk(tf_dict, idf_dict, k)
	return tfidf_topk
	
def get_word_probs(sample):
	tf_dict = get_tf(sample)
	prob ={}
	num = float(len(sample))
	for i in tf_dict.keys():
		prob[i] = tf_dict[i] / num
	return prob
	
def cosine_sim(l1, l2):
	num = [l1[i]*l2[i] for i in range(0,len(l1))]
	num = sum(num)
	den1 = [i*i for i in l1]
	den2 = [i*i for i in l2]
	den = math.sqrt(sum(den1)) * math.sqrt(sum(den2))
	return num/den
	
def vectorize_tfidf(feature_space, idf_dict, sample):
	#tf_dict = get_tf(sample)
	#tfidf_dict = get_tfidf(tf_dict, idf_dict)
	vector = []
	for i in feature_space.keys():
		tfidf = sample.count(i)*idf_dict[i]
		vector.insert(feature_space[i],tfidf)
	return vector
	
def get_section_representations(dirname, idf_dict, feature_space):
	file_list = get_all_files(dirname)
	section_reprs = {}
	for file in file_list:
		sample = flatten(load_file_excerpts(file))
		file = os.path.basename(file)
		file = os.path.splitext(file)[0]
		vector = vectorize_tfidf(feature_space, idf_dict, sample)
		section_reprs[file] = vector
	return section_reprs
	
def predict_class(excerpt, representation_dict, feature_space, idf_dict):
	cosine_similarities = {}
	sample = standardize(excerpt)
	fileObject = open("idf-pickle1","r")
        idf_dict = pickle.load(fileObject)
        fileObject.close()
	vector = vectorize_tfidf(feature_space, idf_dict, sample)
	for i in representation_dict.keys():
		cosine_similarities[i] = cosine_sim(vector, representation_dict[i])
	predicted_class = max(cosine_similarities, key = cosine_similarities.get)
	return predicted_class
	
def label_sents(excerptfile):
	corpus = load_directory_excerpts('project_articles_train/')
	sample = flatten(corpus)
	labels = []
	tfidfk = get_tfidf_topk(sample, corpus, 1500)
	fileObject = open("idf-pickle1","r")
	idf_dict = pickle.load(fileObject)
	fileObject.close()
	wordlist = (dict(tfidfk)).keys()
	feat_dict = create_feature_space(wordlist)
	repr_dict = get_section_representations('project_articles_train/', idf_dict, feat_dict)
	for line in excerptfile.readlines():
		class_label = predict_class(line, repr_dict, feat_dict, idf_dict)
		if(str(class_label) == "gina_excerpts"):
			labels.append(1)
		elif(str(class_label) == "non_gina_excerpts"):
			labels.append(0)
	excerptfile.close()
	return labels
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
		
def create_vocab():
	corpus = load_directory_excerpts('project_articles_train/')
	sample = flatten(corpus)
	words = []
	tfidfk = get_tfidf_topk(sample, corpus, 2000)
	for i in tfidfk:
		words.append(i[0])
	return words
	
def get_words(paragraph):
	sentences = []
	words = []
	#stemmer = PorterStemmer()
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
		#freq_dict = Counter(new_words)
		fileObject = open("idf-pickle1","r")
        	idf_dict = pickle.load(fileObject)
        	fileObject.close()
		for w in new_words:
			pair = str(fs[w]) + ":" + str(idf_dict[w])
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
	vocab = create_vocab()
	X = np.array(X_train)^M
        Y = np.array(Y_train)^M
        accuracies = []^M
        kf = StratifiedKFold(Y, n_folds=10)^M
        for train_index, test_index in kf:
                train_X, train_Y = X[train_index], Y[train_index]^M
                test_X, test_Y = X[test_index], Y[test_index] ^M
                vocab = create_vocab()^M
                convert_to_svm(train_X, train_Y, 'train.svm', vocab)^M
                convert_to_svm(test_X, test_Y, 'test.svm', vocab)^M
                labels, accuracy, values = train_test_model('train.svm', 'test.svm')^M
                accuracies.append(float(accuracy[0]))^M
        print accuracies^M
        acc = sum(accuracies)/len(accuracies)^M
        return  acc
	
if __name__ == "__main__": 
	acc = predict()
	print acc
