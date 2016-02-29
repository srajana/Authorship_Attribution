from __future__ import division
from nltk import word_tokenize
from collections import Counter
import os
import glob
import math
import pickle

def get_all_files(directory):
	files = []
	for name in os.listdir(directory):
		if os.path.isfile(os.path.join(directory, name)):
			files.append(os.path.join(directory, name))
	return files

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
	
def get_mi(sample, corpus):
	mi_dict = {}
	pw_sample = get_word_probs(sample)
	corpus = flatten(corpus)
	n = corpus.__len__()
	sample_set = set(sample)
	for word in sample_set:
		count = corpus.count(word)
		if count >=5:
			pw_corpus = count/float(n)
			mi_dict[word] = math.log(pw_sample[word] / pw_corpus)
	return mi_dict
	
def get_mi_topk(sample, corpus, k):
	mi_dict = get_mi(sample, corpus)
	rank_list = sorted(mi_dict, key=mi_dict.__getitem__, reverse=True)
	mi_topk = []
	for i in range(0,k):
		mi_topk.append((rank_list[i], mi_dict[rank_list[i]]))
	return mi_topk
	
def get_precision(l1, l2):
	count_common = 0
	count_sample = float(len(l1))
	for i in l1:
		if i in l2:
			count_common = count_common + 1
	precision = count_common / count_sample
	return precision
	
def get_recall(l1, l2):
	count_common = 0
	count_ref = float(len(l2))
	for i in l1:
		if i in l2:
			count_common = count_common + 1
	recall = count_common / count_ref
	return recall
	
def cosine_sim(l1, l2):
	num = [l1[i]*l2[i] for i in range(0,len(l1))]
	num = sum(num)
	den1 = [i*i for i in l1]
	den2 = [i*i for i in l2]
	den = math.sqrt(sum(den1)) * math.sqrt(sum(den2))
	return num/den
	
def create_feature_space(wordlist):
	#unique_list = list(set(wordlist))
	feature_space = {}
	for i in range(0,len(wordlist)):
		feature_space[wordlist[i]] = i
	return feature_space
	
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
	vector = vectorize_tfidf(feature_space, idf_dict, sample)
	for i in representation_dict.keys():
		cosine_similarities[i] = cosine_sim(vector, representation_dict[i])
	predicted_class = max(cosine_similarities, key = cosine_similarities.get)
	return predicted_class
	
def label_sents(excerptfile, outputfile):
	corpus = load_directory_excerpts('/home1/c/cis530/hw1/data/train/')
	sample = flatten(corpus)
	tfidfk = get_tfidf_topk(sample, corpus, 1000)
	fileObject = open("idf-pickle1","r")
	idf_dict = pickle.load(fileObject)
	fileObject.close()
	wordlist = (dict(tfidfk)).keys()
	feat_dict = create_feature_space(wordlist)
	repr_dict = get_section_representations('/home1/c/cis530/hw1/data/train/', idf_dict, feat_dict)
	for line in excerptfile.readlines():
		class_label = predict_class(line, repr_dict, feat_dict, idf_dict)
		outputfile.write(str(class_label) + "\n")
	excerptfile.close()
	outputfile.close()
	
def prepare_cluto_tfidf(samplefile2, labelfile, matfile, corpus):
	with open(samplefile2,'r') as f:
		num_lines = sum(1 for line in f)
	labelfile = open(labelfile, "w")
	matfile = open(matfile, "w")
	tfidf_topk = get_tfidf_topk(flatten(corpus), corpus, 1000)
	wordlist = (dict(tfidf_topk)).keys()
	feature_space = create_feature_space(wordlist)
	fileObject = open("idf-pickle1","r")
	idf_dict = pickle.load(fileObject)
	fileObject.close()
	matfile.write(str(num_lines)+ " " + str(len(wordlist)) + "\n")
	for i in wordlist:
		labelfile.write(str(i) + "\n")
	with open(samplefile2,'r') as f:
		for line in f:
			sample = standardize(line)
			vector = vectorize_tfidf(feature_space, idf_dict, sample)
			vector = ' '.join(map(str,vector))
			matfile.write(vector + "\n")
	labelfile.close()
	matfile.close()
	
def prepare_cluto_mi(samplefile2, labelfile, matfile, corpus):
	with open(samplefile2,'r') as f:
		num_lines = sum(1 for line in f)
	labelfile = open(labelfile, "w")
	matfile = open(matfile, "w")
	tfidf_topk = get_tfidf_topk(flatten(corpus), corpus, 1000)
	wordlist = (dict(tfidf_topk)).keys()
	feature_space = create_feature_space(wordlist)
	matfile.write(str(num_lines)+ " " + str(len(wordlist)) + "\n")
	word_prob = get_word_probs(flatten(corpus))
	for i in wordlist:
		labelfile.write(str(i) + "\n")
	with open(samplefile2,'r') as f:
		for line in f:
			sample = standardize(line)
			vector = vectorize_mi(feature_space, word_prob, sample)
			vector = ' '.join(map(str,vector))
			matfile.write(vector + "\n")
	labelfile.close()
	matfile.close()
	
def generate_mi_feature_labels(dirname, k, corpus):
	words = []
	file_list = get_all_files(dirname)
	for file in file_list:
		sample = flatten(load_file_excerpts(file))
		mitopk = get_mi_topk(sample, corpus, k)
		words.extend(dict(mitopk).keys())
	return words
	
def vectorize_mi(feature_space, word_probs, sample):
	vector = []
	mi_dict = {}
	pw_sample = get_word_probs(sample)
	for word in feature_space.keys():
		if word not in pw_sample.keys():
			mi = 0
		else:
			mi = math.log(pw_sample[word] / word_probs[word])
		vector.insert(feature_space[word],mi)
	return vector
	
if __name__=="__main__":
	sample = load_file_excerpts('/home1/c/cis530/hw1/data/train/background.txt')
	sample = flatten(sample)
	corpus = load_directory_excerpts('/home1/c/cis530/hw1/data/train/')
	
	#2.1 - TF_IDF
	tfidf_top_k = get_tfidf_topk(sample, corpus, 1000)
	tfidf_topk = flatten(tfidf_top_k)
	fileObject = open("idf-pickle1","r")
	idf_dict = pickle.load(fileObject) 
	fileObject.close()
	file1 = open("hw1_2-1a.txt","w")
	file2 = open("hw1_2-1b.txt", "w")
	for i,k in zip(tfidf_topk[0::2],tfidf_topk[1::2]):
		file1.write(str(i) + "\t" + str(k) + "\n")
		file2.write(str(i) + "\t" + str(idf_dict[i]) + "\n")
	file1.close()
	file2.close()
	
	#2.2 - MI
	mi_top_k = get_mi_topk(sample, corpus, 1000)
	mi_topk = flatten(mi_top_k)
	file3 = open("hw1_2-2.txt","w")
	for i,k in zip(mi_topk[0::2],mi_topk[1::2]):
		file3.write(str(i) + "\t" + str(k) + "\n")
	file3.close()
	
	#2.3 - Precision and Recall
	L_sample = (dict(mi_top_k)).keys()
	L_ref = (dict(tfidf_top_k)).keys()
	precision = get_precision(L_sample, L_ref)
	recall = get_recall(L_sample, L_ref)
	file4 = open("writeup.txt","w")
	file4.write("Precision: " + str(precision) + "\n")
	file4.write("Recall: " + str(recall) + "\n")
	file4.close()

	#4 - Class prediction
	file5 = open("/home1/c/cis530/hw1/data/test/samples.txt", "r")
	file6 = open("hw1_4.txt", "w")
	label_sents(file5, file6)
	
	#5 - CLUTO
	corpus = load_directory_excerpts('/home1/c/cis530/hw1/data/train/')
	prepare_cluto_tfidf('/home1/c/cis530/hw1/data/clustering/excerpts.txt', 'labelfile_tfidf.txt', 'matfile_tfidf.txt', corpus)
	prepare_cluto_mi('/home1/c/cis530/hw1/data/clustering/excerpts.txt', 'labelfile_mi.txt', 'matfile_mi.txt', corpus)
