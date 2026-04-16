#!/usr/bin/python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import unicodedata
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack, vstack
import random
import pickle


def load_plot_data(filename='plot.list'):
    f = open(filename, 'rb')
    line = f.readline()
    lookup = {}

    cur_movie = ""
    while line != '':
        comps = line.split(':')
        if len(comps) > 1:
            if comps[0] == 'MV':
                cur_movie = clean_title(':'.join(comps[1:]))

            if cur_movie not in lookup:
                lookup[cur_movie] = ""

            if comps[0] == 'PL':
                lookup[cur_movie] = lookup[cur_movie] + ' ' + ':'.join(comps[1:]).rstrip()

        line = f.readline()
    return lookup


def load_tag_data(filename='imdb-genres.list'):
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
                title = ' '.join(comps[0:len(comps) - 1])
                realtitle = clean_title(title)

                if realtitle not in lookup:
                    lookup[realtitle] = [comps[-1].rstrip()]
                else:
                    lookup[realtitle].append(comps[-1].rstrip())

        line = f.readline()
    return lookup


def join_plot_tags(plot, tags):
    join_keys = [k for k in plot if k in tags]
    result = []
    for i in join_keys:
        result.append((i, plot[i], tags[i]))
    return result


def data_to_features(data, label1='Comedy', label2='Horror'):
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english', max_features=50000)
    # vectorizer = CountVectorizer(min_df=1, max_features=10000)

    text = [unicodedata.normalize('NFKD', unicode(i[0] + " " + i[1], errors='replace')).encode('ascii', 'ignore') for i
            in data]
    X = vectorizer.fit_transform(text)
    Y = []
    index = []
    inc = 0

    for i in data:
        tagstring = ' '.join(i[2])
        # print tagstring, i[2]
        if label2 in tagstring:
            Y.append(0)
            index.append(inc)
        elif label1 in tagstring:
            # print tagstring
            Y.append(1)
            index.append(inc)

        inc = inc + 1

    return X[index, :], np.array(Y), index


def full_data_to_features(data):
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english', max_features=50000)
    # vectorizer = CountVectorizer(min_df=1, max_features=10000)

    text = [unicodedata.normalize('NFKD', unicode(i[0] + " " + i[1], errors='replace')).encode('ascii', 'ignore') for i
            in data]
    X = vectorizer.fit_transform(text)
    return X


def clean_title(title):
    realtitlec = title.split('(')[0]
    realtitle = realtitlec.strip('"')
    realtitle = realtitle.strip()

    # deal with the empty string problem
    if len(realtitle) == 0:
        realtitle = 'N\A' + str(random.random())

    return realtitle.strip().lower()


def load_yahoo_data(filename='ydata-ymovies-movie-content-descr-v1_0.txt'):
    f = open(filename, 'rb')
    line = f.readline()
    dataset = {}
    while line != '':
        comps = line.split('\t')
        dataset[clean_title(comps[1])] = (comps[2], comps[10].split('|'))
        line = f.readline()
    return dataset


# lookup movie tuple
def clean(movie, yahoo):
    title = movie[0]

    if title in yahoo and yahoo[title][1][0] != '\\N':
        clean = yahoo[title]
        # print 'Title', title, 'found in yahoo', clean[1], movie[2]
        return (title, title + " " + movie[1] + " " + clean[0], clean[1])
    else:
        # remove for now
        if len(movie[2]) > 5:
            # print 'Removed tag', movie[0]
            return (title, movie[1], ['\\N'])

    return movie


def sampleclean(dirty_data, clean_data, test_data, indextuple, batchsize=50, total=10000):
    X_clean = clean_data[0]
    y_clean = clean_data[1]
    X_test = test_data[0]
    y_test = test_data[1]

    dirtyex = [i for i in indextuple[0]]
    cleanex = []

    for i in range(0, total, batchsize):
        clf = SGDClassifier(loss="hinge", alpha=0.00001, n_iter=200, fit_intercept=True, warm_start=False)
        # uniform sampling
        examples_real = np.random.choice(dirtyex, batchsize, replace=False)
        examples_map = translate_indices(examples_real, indextuple[2])
        cleanex.extend(examples_map)

        for j in examples_real:
            dirtyex.remove(j)

        clf.fit(X_clean[cleanex, :], y_clean[cleanex])
        print
        "[SampleClean] Number Cleaned So Far ", len(cleanex)
        ypred = clf.predict(X_test)
        print
        "[SampleClean] Prediction Freqs", np.sum(ypred), np.shape(ypred)
        print
        classification_report(y_test, ypred)

        print
        "[SampleClean] Accuracy ", i, accuracy_score(y_test, ypred)


def activeclean(dirty_data, clean_data, test_data, full_data, indextuple, batchsize=50, total=10000):
    # makes sure the initialization uses the training data
    X = dirty_data[0][translate_indices(indextuple[0], indextuple[1]), :]
    y = dirty_data[1][translate_indices(indextuple[0], indextuple[1])]

    X_clean = clean_data[0]
    y_clean = clean_data[1]

    X_test = test_data[0]
    y_test = test_data[1]

    print
    "[ActiveClean Real] Initialization"

    lset = set(indextuple[2])
    dirtyex = [i for i in indextuple[0]]
    cleanex = []

    total_labels = []

    ##Not in the paper but this initialization seems to work better, do a smarter initialization than
    ##just random sampling (use random initialization)
    topbatch = np.random.choice(range(0, len(dirtyex)), batchsize)
    examples_real = [dirtyex[j] for j in topbatch]
    examples_map = translate_indices(examples_real, indextuple[2])

    # Apply Cleaning to the Initial Batch
    cleanex.extend(examples_map)
    for j in examples_real:
        dirtyex.remove(j)

    clf = SGDClassifier(loss="hinge", alpha=0.000001, n_iter=200, fit_intercept=True, warm_start=True)
    clf.fit(X_clean[cleanex, :], y_clean[cleanex])

    for i in range(50, total, batchsize):
        print
        "[ActiveClean Real] Number Cleaned So Far ", len(cleanex)
        ypred = clf.predict(X_test)
        print
        "[ActiveClean Real] Prediction Freqs", np.sum(ypred), np.shape(ypred)
        print
        classification_report(y_test, ypred)

        # Sample a new batch of data
        examples_real = np.random.choice(dirtyex, batchsize)
        examples_map = translate_indices(examples_real, indextuple[2])

        total_labels.extend([(r, (r in lset)) for r in examples_real])

        # on prev. cleaned data train error classifier
        ec = error_classifier(total_labels, full_data)

        for j in examples_real:
            try:
                dirtyex.remove(j)
            except ValueError:
                pass

        dirtyex = ec_filter(dirtyex, full_data, ec)

        # Add Clean Data to The Dataset
        cleanex.extend(examples_map)

        # uses partial fit (not in the paper--not exactly SGD)
        clf.partial_fit(X_clean[cleanex, :], y_clean[cleanex])

        print
        "[ActiveClean Real] Accuracy ", i, accuracy_score(y_test, ypred)

        if len(dirtyex) < 50:
            print
            "[ActiveClean Real] No More Dirty Data Detected"
            break;


def translate_indices(globali, imap):
    lset = set(globali)
    return [s for s, t in enumerate(imap) if t in lset]


def error_classifier(total_labels, full_data):
    indices = [i[0] for i in total_labels]
    labels = [int(i[1]) for i in total_labels]
    if np.sum(labels) < len(labels):
        clf = SGDClassifier(loss="log", alpha=1e-6, n_iter=200, fit_intercept=True)
        clf.fit(full_data[indices, :], labels)
        # print labels
        # print clf.score(full_data[indices,:],labels)
        return clf
    else:
        return None


def ec_filter(dirtyex, full_data, clf, t=0.90):
    if clf != None:
        pred = clf.predict_proba(full_data[dirtyex, :])
        # print pred
        # print len([j for i,j in enumerate(dirtyex) if pred[i][0] < t]), len(dirtyex)
        return [j for i, j in enumerate(dirtyex) if pred[i][0] < t]

    print
    "CLF none"

    return dirtyex


def retraining(dirty_data, clean_data, test_data, full_data, indextuple, batchsize=50, total=10000):
    # makes sure the initialization uses the training data
    X = dirty_data[0][translate_indices(indextuple[0], indextuple[1]), :]
    y = dirty_data[1][translate_indices(indextuple[0], indextuple[1])]

    X_clean = clean_data[0]
    y_clean = clean_data[1]

    X_test = test_data[0]
    y_test = test_data[1]

    lset = set(indextuple[2])
    dirtyex = [i for i in indextuple[0]]
    cleanex = []

    for i in range(batchsize, total, batchsize):
        examples_real = np.random.choice(dirtyex, batchsize, replace=False)
        examples_map = translate_indices(examples_real, indextuple[2])
        cleanex.extend(examples_map)

        for j in examples_real:
            dirtyex.remove(j)

        indicesdirty = translate_indices(dirtyex, indextuple[1])

        # print np.shape(X_clean[cleanex,:]), np.shape(dirty_data[0][indicesdirty,:])
        X_rt = vstack((dirty_data[0][indicesdirty, :], X_clean[cleanex, :]))
        # print X_rt
        y_rt = np.hstack((np.squeeze(dirty_data[1][indicesdirty]), np.squeeze(y_clean[cleanex])))
        # print np.shape(X_rt), np.shape(y_rt)

        clf = SGDClassifier(loss="hinge", alpha=0.000001, n_iter=200, fit_intercept=True)
        clf.fit(X_rt, y_rt)
        print
        "[Retraining] Number Cleaned So Far ", len(cleanex)
        ypred = clf.predict(X_test)
        print
        "[Retraining] Prediction Freqs", np.sum(ypred), np.shape(ypred)
        print
        classification_report(y_test, ypred)
        print
        "[Retraining] Accuracy ", i, accuracy_score(y_test, ypred)


data = pickle.load(open('imdb_features.p', 'rb'))

X_clean = data['X_clean']
y_clean = data['y_clean']
X_dirty = data['X_dirty']
y_dirty = data['y_dirty']
X_full = data['X_full']
indices_dirty = data['indices_dirty']
indices_clean = data['indices_clean']
size = data['shape']
examples = np.arange(0, size, 1)
train_indices, test_indices = train_test_split(examples, test_size=0.20)
clean_test_indices = translate_indices(test_indices, indices_clean)

activeclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (X_clean[clean_test_indices, :], y_clean[clean_test_indices]),
            X_full,
            (train_indices, indices_dirty, indices_clean))

"""


retraining((X_dirty, y_dirty),
				 (X_clean, y_clean),
				 (X_clean[clean_test_indices,:], y_clean[clean_test_indices]),
				 X_full,
				 (train_indices,indices_dirty,indices_clean))
"""

sampleclean((X_dirty, y_dirty),
            (X_clean, y_clean),
            (X_clean[clean_test_indices, :], y_clean[clean_test_indices]),
            (train_indices, indices_dirty, indices_clean))

