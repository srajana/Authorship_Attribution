from __future__ import division
import math
import os
import glob
import subprocess
from nltk import sent_tokenize
from nltk import word_tokenize
from collections import Counter

def srilm_preprocess(raw_text, temp_file):
	sentences = sent_tokenize(raw_text)
	f = open(temp_file, "w")
	for sent in sentences:
		f.write(sent.encode('utf8') + "\n")
	f.close()
		
def srilm_bigram_models(input_file, output_dir):
	input = os.path.basename(input_file)
	with open(input_file, "r") as f:
		data = f.read().decode('utf8')
	srilm_preprocess(data, 'temp_srajana.txt')
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	model_file1 = str(output_dir) + "/" + str(input) + ".uni.lm"
	model_file11 = str(output_dir) + "/" + str(input) + ".uni.counts"
	model_file2 = str(output_dir) + "/" + str(input) + ".bi.lm"
	model_file3 = str(output_dir) + "/" + str(input) + ".bi.kn.lm"
	#subprocess.call(['/home1/c/cis530/srilm/ngram-count', '-text', 'temp_srajana.txt', '-lm', model_file1, '-order', '1', '-addsmooth', '0.25', '-write', model_file11])
	subprocess.call(['/home1/c/cis530/srilm/ngram-count', '-text', 'temp_srajana.txt', '-lm', model_file2, '-order', '7', '-addsmooth', '0.25'])
	#subprocess.call(['/home1/c/cis530/srilm/ngram-count', '-text', 'temp_srajana.txt', '-lm', model_file3, '-order', '5', '-kndiscount'])
		
def srilm_ppl(model_file, raw_text):
	srilm_preprocess(raw_text, 'temp2_srajana.txt')
	ppl = subprocess.check_output(['/home1/c/cis530/srilm/ngram', '-lm', model_file, '-ppl', 'temp2_srajana.txt'])
	index = ppl.find('ppl=')
	value = ppl[index:]
	perp = value.split("=")[1].split(" ")[1]
	return float(perp)
	
def lm_predict(models, test_file, pred_file):
	f1 = open(test_file, "r")
	#f2 = open(pred_file, "w")
	labels = []
	for line in f1.readlines():
		perplexities = {}
		line = line.decode('utf8')
		for m in models:
			ppl = srilm_ppl(m[1], line)
			perplexities[m[0]] = ppl
		min_value = min(perplexities.itervalues())
		min_keys = [k for k in perplexities if perplexities[k] == min_value]
		predicted = min(min_keys)
		if str(predicted) == “gina_excerpts”:
			labels.append(1)
		elif str(predicted) == “non_gina_excerpts”:
			labels.append(0)
		#f2.write(str(predicted) + "\n")
	f1.close()
	#f2.close()
	return labels
	
def create_feature_space(wordlist):
	feature_space = {}
	for i in range(0,len(wordlist)):
		feature_space[wordlist[i]] = i
	return feature_space

def predict():
	‘’’trainfiles = ['gina_excerpts.txt','non_gina_excerpts.txt']
	
	for file in trainfiles:
		srilm_bigram_models(file, 'CIS530-srajana’)’’’

	srilm_bigram_models(‘gina_excerpts.txt’, ‘CIS530-srajana’)
	
	#models = [('gina_excerpts', 'CIS530-srajana/gina_excerpts.txt.bi.kn.lm'), ('non_gina_excerpts', 'CIS530-srajana/non_gina_excerpts.txt.bi.kn.lm')]
	models = [('gina_excerpts', 'CIS530-srajana/gina_excerpts.txt.bi.kn.lm')]
	labels = lm_predict(models,'/home1/c/cis530/project/data/project_articles_test','labels_lm_kl.txt')
	return labels
	
if __name__ == "__main__":
	labels = predict()
	
		