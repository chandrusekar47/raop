import csv
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def read_embeddings(filename):
	with open(filename, 'r') as f:
		data = list(csv.reader(f))
	embeddings = {}
	for d in data:
		embeddings[d[0]] = [float(x) for x in d[1:]]
	return embeddings

def read_with_labels(filename):
	with open(filename, 'r') as f:
		data = list(csv.reader(f))
	labels = data.pop(0)
	return data, labels

def plot(vecs, filename='raop.png'):

	plt.figure(figsize=(18, 18))
	for v in vecs:
		x, y = v[2]
		col = 'g' if v[1] == 1 else 'r'
		plt.scatter(x, y, color=col, alpha=0.5)
		#plt.annotate(v[0], xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

	win = patches.Patch(color='g', label = 'Success')
	lose = patches.Patch(color='r', label = 'Failure')
	plt.figlegend(handles=[win, lose], labels=('Success', 'Failure'), loc='lower center', ncol=2)
	plt.savefig(filename)
	


data, labels = read_with_labels('data/filtered_features_1.csv')
embeddings = read_embeddings('data/raop_embeddings.csv')

vecLen = len(embeddings['UNK'])

titleIdx = labels.index('title')
requestIdx = labels.index('edited_text')
successIdx = labels.index('success')
idIdx = labels.index('request_id')

vectors = []
for d in data:
	title = d[titleIdx]
	request = d[requestIdx]
	success = int(d[successIdx])
	postId = d[idIdx]

	reqVec = [0 for _ in range(vecLen)]
	for w in request.split(' '):
		if w in embeddings:
			reqVec = [sum(x) for x in zip(reqVec, embeddings[w])]
		else:
			reqVec = [sum(x) for x in zip(reqVec, embeddings['UNK'])]

	titleVec = [0 for _ in range(vecLen)]
	for w in title.split(' '):
		if w in embeddings:
			titleVec = [sum(x) for x in zip(titleVec, embeddings[w])]

	vectors.append([postId, success, titleVec, reqVec])

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plotOnly = 500
ids = [v[0] for v in vectors[:plotOnly]]
successes = [v[1] for v in vectors[:plotOnly]]
lowDimReqVecs = tsne.fit_transform(np.array([v[3] for v in vectors])[:plotOnly, :])
lowDimTitleVecs = tsne.fit_transform(np.array([v[2] for v in vectors])[:plotOnly, :])
finalReqVecs = zip(ids, successes, lowDimReqVecs)
finalTitleVecs = zip(ids, successes, lowDimTitleVecs)
plot(finalReqVecs, 'raop-req.png')
plot(finalTitleVecs, 'raop-title.png')
