import project_top_1700_stemmed as stemmed1
import project_top_2000_stemmed as stemmed2
import lang_models as lm

labels1 = stemmed1.predict()
labels2 = stemmed2.predict()
labels3 = lm.predict()
new_labels = []

if len(labels1) == len(labels2) == len(labels3):
	for i in range(len(labels1)):
		if(type(labels1[i])!=type(labels2[i])!=type(lables3[i])):
			print "type-mismatch"
			break
		if(labels1[i] != labels2[i]):
			new_labels.append(labels3[i])
		else:
			new_labels.append(labels1[i])
	f = open(“test.txt","w")
	for l in new_labels:
		f.write(str(l) + "\n")
	f.close()
else:
	print "Lengths don't match"
