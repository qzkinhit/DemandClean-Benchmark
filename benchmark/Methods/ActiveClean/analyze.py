#!/usr/bin/python

from sklearn.feature_extraction.text import TfidfVectorizer
import unicodedata
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import random

def load_y_data(filename='ydata-ymovies-movie-content-descr-v1_0.txt'):
	f = open(filename, 'rb')
	line = f.readline()
	dataset = []
	while line != '':
		comps = line.split('\t')
		dataset.append((comps[1], comps[2], comps[10].split('|')))
		line = f.readline()
	return dataset

def load_i_data(filename='imdb-genres.list', label=11):
	f = open(filename, 'rb')
	line = f.readline()
	start = False
	lookup = {}

	while line != '':
		if '==================' in line:
			start = True

		if start:
			comps = line.split('\t')
			if len(comps) > 2:
				title = ' '.join(comps[0:len(comps)-1])
				realtitlec = title.split('(')[0]
				realtitle = realtitlec.strip('"')
				realtitle = realtitle.rstrip().lower()

				if realtitle not in lookup:
					lookup[realtitle] = [comps[-1].rstrip()]
				else:
					lookup[realtitle].append(comps[-1].rstrip())

		line = f.readline()
	return lookup

def cross_reference(ydata, idata, use=True):

	raw_data = []

	for ymovie in ydata:
		realtitle = ymovie[0].split('(')[0].lower()

		if use and realtitle in idata:
			l = idata[realtitle]
			l.extend(ymovie[2])
			raw_data.append((realtitle, ymovie[1], l))
		else:
			raw_data.append((realtitle,ymovie[1], ymovie[2]))

	random.shuffle(raw_data)
	return raw_data

def data_to_features(data, label1='Comedy', label2='Horror'):
	vectorizer = TfidfVectorizer(min_df=1,stop_words='english')
	text = [unicodedata.normalize('NFKD', unicode(i[0] + " " + i[1],errors='replace')).encode('ascii','ignore') for i in data]
	X = vectorizer.fit_transform(text)
	Y = []
	index = []
	inc = 0
	for i in data:
		tagstring = ' '.join(i[2])
		#print tagstring, i[2]
		if label1 in tagstring:
			print tagstring
			Y.append(1)
			index.append(inc)
		elif label2 in tagstring:
			Y.append(0)
			index.append(inc)

		inc = inc + 1

	return X[index,:], np.array(Y)


print '---Without Data Cleaning---'
ydata = load_y_data()
idata = load_i_data()
data = cross_reference(ydata,idata)
X,y = data_to_features(data)
print np.sum(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
clf = SVC(C=100, kernel='linear')
clf.fit(X_train, y_train)
#print clf.coef_[0]

ypred = clf.predict(X_test)
print np.sum(ypred), np.shape(ypred)
print  classification_report(y_test, ypred)


print '---With Data Cleaning---'
data = cross_reference(ydata,idata, False)
X,y = data_to_features(data)
print np.sum(y)

X_train = X[0:7700,:]
y_train = y[0:7700]

clf = SVC(C=100, kernel='linear')
clf.fit(X_train, y_train)

data = cross_reference(ydata,idata)
X,y = data_to_features(data)
X_test = X[7700:,:]
y_test = y[7700:]

ypred = clf.predict(X_test)
print np.sum(ypred), np.shape(ypred)
print  classification_report(y_test, ypred)
