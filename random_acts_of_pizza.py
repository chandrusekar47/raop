from __future__ import print_function
import json
import re
import csv
import numpy as np
import nltk
import datetime
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.mixture import GaussianMixture
from collections import Counter
import sklearn.preprocessing
import keras.models
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.backend as K
from tfidf_embedding_vectorizer import *
from keras import optimizers
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import svm


Word2VecModel = {}
sentiment_analyzer = SentimentIntensityAnalyzer()
vectorizer = CountVectorizer(ngram_range =(3,3) , analyzer = "word", tokenizer = nltk.word_tokenize, preprocessor = None, stop_words = stopwords.words('english'), max_features = 10000)
Training_bag_of_words_features = []
Testing_bag_of_words_features = []
selected_features = [1,2,4,6,7,8,14,16,17,18,27]

headers = ["request_id",
	"acct_age_in_days",
	"days_since_first_post_on_raop",
	"acct_no_of_comments",
	"acct_no_of_comments_in_raop",
	"acct_no_of_posts",
	"acct_no_of_posts_in_raop",
	"no_of_subreddits_posted",
	"request_month",
	"request_day_of_year",
	"request_day_of_month",
	"request_hour_of_day",
	"no_of_votes",
	"post_karma",
	"pos_score",
	"neg_score",
	"neutral_score",
	"no_words_title",
	"no_words_posts",
	"title_length",
	"post_length",
	"pos_words_percent",
	"neg_words_percent",
	"neutral_words_percent",
	"post_adj_words_percent",
	"title_adj_words_percent",
	"subreddits_posted",
	"title",
	"edited_text",
	"narrative",
	"success"]
selected_features = map(headers.index, ["days_since_first_post_on_raop","no_words_posts","post_length","acct_no_of_comments_in_raop","acct_no_of_posts_in_raop", "pos_score", "neg_score", "request_month", "pos_words_percent", "neg_words_percent", "request_day_of_year"])
# todo: change this have the name of the feature like the previous line
selected_features = [1,2,4,6,7,8,14,16,17,18]

narratives = ['money', 'job', 'student', 'family', 'craving']
triggers = [
	['money', 'now', 'broke', 'week', 'until', 'time', 'last', 'day', 'when', 'today', 'tonight', 'paid', 'next', 'first', 'night', 'after', 'tomorrow', 'month', 'while', 'account', 'before', 'long', 'Friday', 'rent', 'buy', 'bank', 'still', 'bills', 'bills', 'ago', 'cash', 'due', 'due', 'soon', 'past', 'never', 'paycheck', 'check', 'spent', 'years', 'poor', 'till', 'yesterday', 'morning', 'dollars', 'financial', 'hour', 'bill', 'evening', 'credit', 'budget', 'loan', 'bucks', 'deposit', 'dollar', 'current', 'payed'],
	['work', 'job', 'paycheck', 'unemployment', 'interview', 'fired', 'employment', 'hired', 'hire'],
	['college', 'student', 'school', 'roommate', 'studying', 'university', 'finals', 'semester', 'class', 'study', 'project', 'dorm', 'tuition'],
	['family', 'mom', 'wife', 'parents', 'mother', 'husband', 'dad', 'son', 'daughter', 'father', 'parent', 'mum'],
	['friend', 'girlfriend', 'craving', 'birthday', 'boyfriend', 'celebrate', 'party', 'game', 'games', 'movie', 'date', 'drunk', 'beer', 'celebrating', 'invited', 'drinks', 'crave', 'wasted', 'invite']
]

def cleanup(string):
	if string == None:
		return ""
	string = re.sub("[^a-zA-Z\d\s]", " ", string)
	string = string.replace("\n", " ").replace("\r", "").replace("\t", " ").strip(" ")
	string = re.sub("\s{2,}", " ", string)
	return string

def array_to_str(array):
	return u','.join(map(lambda x: cleanup(x), array)).encode('utf-8')

def month_of_year(unix_timestamp):
	return int(datetime.datetime.fromtimestamp(int(unix_timestamp)).strftime('%m'))

def day_of_year(unix_timestamp):
	return int(datetime.datetime.fromtimestamp(int(unix_timestamp)).strftime('%j'))

def day_of_month(unix_timestamp):
	return int(datetime.datetime.fromtimestamp(int(unix_timestamp)).strftime('%d'))

def hour_of_day(unix_timestamp):
	return int(datetime.datetime.fromtimestamp(int(unix_timestamp)).strftime('%H'))

def num_words(text):
	return float(len(text.split(" ")))

def group_words_by_sentiment(text):
	words = text.split()
	word_scores = zip(words, map(sentiment_analyzer.polarity_scores, words))
	pos_words = [x[0] for x in word_scores if x[1]['pos'] >= 0.5]
	neg_words = [x[0] for x in word_scores if x[1]['neg'] >= 0.5]
	neutral_words = [x[0] for x in word_scores if x[1]['neu'] >= 0.5]
	return (pos_words, neg_words, neutral_words)

def get_adjectives(text):
	text = nltk.word_tokenize(text)
	return map(lambda x: x[0], filter(lambda x: x[1] == "JJ", nltk.pos_tag(text)))

def get_narrative(text):
	counts = [0, 0, 0, 0, 0]
	for w in text.split(' '):
		for t in range(len(narratives)):
			if w in triggers[t]:
				counts[t] += 1
	return counts.index(max(counts))

def dict_to_csv(filename, output_file_name):
	with open(filename) as data_file:
		data = json.load(data_file)	
	output_file = open(output_file_name, 'w')
	print(','.join(headers), file = output_file)
	for record in data:
		values = []
		values.append(record["request_id"])
		values.append(float(record["requester_account_age_in_days_at_request"]))
		values.append(float(record["requester_days_since_first_post_on_raop_at_request"]))
		values.append(float(record["requester_number_of_comments_at_request"]))
		values.append(float(record["requester_number_of_comments_in_raop_at_request"]))
		values.append(float(record["requester_number_of_posts_at_request"]))
		values.append(float(record["requester_number_of_posts_on_raop_at_request"]))
		values.append(float(record["requester_number_of_subreddits_at_request"]))
		values.append(month_of_year(float(record["unix_timestamp_of_request_utc"])))
		values.append(day_of_year(float(record["unix_timestamp_of_request_utc"])))
		values.append(day_of_month(float(record["unix_timestamp_of_request_utc"])))
		values.append(hour_of_day(float(record["unix_timestamp_of_request_utc"])))
		values.append(float(record["requester_upvotes_plus_downvotes_at_request"]))
		values.append(float(record["requester_upvotes_minus_downvotes_at_request"]))
		post_text = cleanup(record["request_text_edit_aware"])
		title = cleanup(record["request_title"])
		scores = sentiment_analyzer.polarity_scores(post_text)
		no_words = num_words(post_text)
		(pos_words, neg_words, neutral_words) = group_words_by_sentiment(post_text)
		values.append(scores['pos'])
		values.append(scores['neg'])
		values.append(scores['neu'])
		values.append(num_words(title))
		values.append(no_words)
		values.append(len(title))
		values.append(len(post_text))
		values.append(len(pos_words)/no_words if no_words !=0 else 0)
		values.append(len(neg_words)/no_words if no_words !=0 else 0)
		values.append(len(neutral_words)/no_words if no_words !=0 else 0)
		values.append(len(get_adjectives(post_text))/no_words if no_words !=0 else 0)
		values.append(len(get_adjectives(title))/num_words(title))
		values.append('"'+array_to_str(record["requester_subreddits_at_request"]) +'"')
		values.append('"'+title+ '"')
		values.append('"'+post_text + '"')
		values.append('"' + str(get_narrative(title + ' ' + post_text)) + '"')
		if record.has_key("requester_received_pizza"):
			values.append(1 if bool(record["requester_received_pizza"]) else 0)
		else:
			values.append('0')
		print(u','.join(map(str, values)).encode('utf-8'), file = output_file)
	output_file.close()

def read_lines_from_file(input_file):
	lines = []
	with open(input_file, "rb") as file:
		reader = csv.reader(file, delimiter = ",")
		headers = next(reader, None)
		for line in reader:
			lines.append(line)
	return (lines, headers)

def generate_bag_of_word_features(post_texts):
	return vectorizer.fit_transform(post_texts).toarray()

def generate_test_bag_of_word_features(post_texts):
	return vectorizer.transform(post_texts).toarray()

def train_random_forest_classifier(training_data, n_est=100, use_bag_of_words = False):
	global Training_bag_of_words_features
	vectorizer = CountVectorizer(analyzer = "word", tokenizer = nltk.word_tokenize, preprocessor = None, stop_words = stopwords.words('english'), max_features = 10000)
	forest = RandomForestClassifier(n_estimators=n_est, class_weight = "balanced")
	forest.classes_ = [0, 1]
	if use_bag_of_words:
		if Training_bag_of_words_features == []:
			Training_bag_of_words_features = generate_bag_of_word_features(training_data[:, -2])
		forest = forest.fit(Training_bag_of_words_features, training_data[:, -1].astype('float'))
		scores = cross_val_score(forest, Training_bag_of_words_features, training_data[:, -1].astype('float'), cv = 5)
	else:
		forest = forest.fit(training_data[:, selected_features].astype('float'), training_data[:, -1].astype('float'))
		scores = cross_val_score(forest, training_data[:, selected_features].astype('float'), training_data[:, -1].astype('float'), cv = 5)
	print("Scores gotten using Random Forest (# of estimators="+str(n_est)+")")
	print(scores)
	print(np.mean(scores))
	return forest, np.mean(scores)

def train_decision_tree_classifer(training_data, depth=50):
	dtree = DecisionTreeClassifier(max_depth=depth, min_samples_split=2,class_weight = "balanced")
	dtree.classes_ = [0, 1]
	scores = cross_val_score(dtree, training_data[:, selected_features].astype('float'), training_data[:, -1].astype('float'), cv = 5)
	print("Scores gotten using Decision Tree (max depth="+str(depth)+")")
	print(scores)
	print(np.mean(scores))
	return dtree, np.mean(scores)

def train_extra_Randomized_forest_classifer(training_data, n_est=10):
	randomized = ExtraTreesClassifier(n_estimators=n_est, max_depth=None, min_samples_split=2, class_weight = "balanced")
	randomized.classes_ = [0, 1]
	scores = cross_val_score(randomized, training_data[:, selected_features].astype('float'), training_data[:, -1].astype('float'), cv = 10)
	print("Scores gotten using Extra Randomized Forests (# of estimators="+str(n_est)+")")
	print(scores)
	print(np.mean(scores))
	return randomized, np.mean(scores)

def train_AdaBoost_classifier(training_data, n_est):
	adaboost = AdaBoostClassifier(n_estimators=n_est)
	adaboost.classes_ = [0, 1]
	scores = cross_val_score(adaboost, training_data[:, selected_features].astype('float'), training_data[:, -1].astype('float'), cv = 10)
	adaboost = adaboost.fit(training_data[:, selected_features].astype('float'), training_data[:, -1].astype('float'))
	print("Scores gotten using AdaBoost classifier (# of estimators="+str(n_est)+")")
	print(scores)
	print(np.mean(scores))
	return adaboost, np.mean(scores)

def train_ensemble_classifier(training_data,forest, dtree, adaboost, extra_random, gnb, regression):
	ensemble = VotingClassifier(estimators=[('rf', forest), ('dt', dtree), ('et',extra_random),('gnb',gnb),('lr',regression)], voting='hard')
	ensemble.classes_ = [0, 1]
	scores = cross_val_score(ensemble, training_data[:, selected_features].astype('float'), training_data[:, -1].astype('float'), cv = 5)
	print("Scores gotten using Ensemble classifier")
	print(str(scores))
	print(np.mean(scores))
	return adaboost

def train_Logistic_regression(training_data):
	x = (training_data[:, selected_features].astype('float'))
	y = (training_data[:, -1].astype('float'))

	clf1 = LogisticRegression(penalty = 'l1', class_weight='balanced')
	clf1 = clf1.fit(x,y)
	#print(headers)
	#print("LOG REGRESSION COEFF : "+str(clf1.coef_[0]) )#acess coefficients
	return clf1

def train_gaussian_NB(training_data):
	x = (training_data[:, selected_features].astype('float'))
	y = (training_data[:, -1].astype('float'))

	gnb = GaussianNB()
	gnb = gnb.fit(x,y)

	return gnb


def generate_submission_file(classifier, submission_filename, use_bag_of_words = False):
	global Testing_bag_of_words_features
	(test_data, _) = read_lines_from_file('data/test_feature_file.csv')
	test_data = np.array(test_data)
	if use_bag_of_words:
		if Testing_bag_of_words_features == []:
			Testing_bag_of_words_features = generate_test_bag_of_word_features(test_data[:, -2])
		output_probabs=classifier.predict_proba(Testing_bag_of_words_features)
	else:
		output_probabs=classifier.predict_proba(test_data[:, selected_features])
	print(np.any(np.array(output_probabs) > 0))
	output_file = open(submission_filename, 'w')
	print("request_id,requester_received_pizza", file = output_file)
	for ind, record in enumerate(test_data):
		print("%s,%f"%(record[0], output_probabs[ind][1]), file = output_file)
	output_file.close()

def generate_feature_files():
	dict_to_csv('data/train.json', 'data/filtered_features.csv')
	dict_to_csv('data/test.json', 'data/test_feature_file.csv')

def generate_normalized_data(training_data):
	num_records = len(training_data)
	dimension = len(training_data[0])
	print("num of records "+str(num_records))
	print("num of features "+str(dimension))

	minmax_scale = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
	for i in range(1,23):
			training_data[:,i] = training_data[:,i].astype(float)			
			training_data[:,i] = minmax_scale.fit_transform(training_data[:,i])

	return training_data

def generate_gaussian_mixture_models(training_data,test_data,submission_filename):
	#training_data = generate_normalized_data(training_data)
	headers = ["request_id",
	"acct_age_in_days",
	"days_since_first_post_on_raop",
	"acct_no_of_comments",
	"acct_no_of_comments_in_raop",
	"acct_no_of_posts",
	"acct_no_of_posts_in_raop",
	"no_of_subreddits_posted",
	"request_month",
	"request_day_of_year",
	"no_of_votes",
	"post_karma",
	"pos_score",
	"neg_score",
	"neutral_score",
	"no_words_title",
	"no_words_posts",
	"title_length",
	"post_length",
	"pos_words_percent",
	"neg_words_percent",
	"neutral_words_percent",
	"post_adj_words_percent",
	"title_adj_words_percent",
	"subreddits_posted",
	"title",
	"edited_text",
	"success"]

	x = (training_data[:, selected_features].astype('float'))
	y = (training_data[:, -1].astype('float'))

	k = 2 #number of clusters/gaussian mixture models
	gmm = GaussianMixture(n_components=k, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10)
	gmm = gmm.fit(x,y)

	xo = (training_data[:,selected_features].astype('float'))
	class_prob = gmm.predict_proba(xo)	
	class_labels = gmm.predict(xo)
	print("Predicted Class labels distribution "+str(Counter(class_labels))+"\n\n")
	output_file = open(submission_filename, 'w')
	y=np.array(y)
	cluster = []
	for c in range(0,k):
		most_likely_class = 99
		temp = []
		c_data = []
		for i in range(0,len(training_data)):
			if class_labels[i] == c:
				temp.append(i)
				features = np.array(selected_features)
				record = np.array(training_data[i][features]).tolist()
				c_data.append(record)
		print("Cluster "+str(c)+" has the following distribution"+str(Counter(y[temp]))+"\n\n")
		most_likely_class = Counter(y[temp]).most_common(1)[0][0]
		cluster.append(c_data)
	
	cluster1 = np.array(cluster)	
	for yo in range(0,k):
		for i in range(0,len(cluster1[yo])):
			cluster1[yo][i] = np.array(cluster1[yo][i]).astype('float')
			cluster1[yo][i] = cluster1[yo][i].tolist()

	#find variances of all those selected dimensions in c_data
	variances = []
	for c in range(0,k):
		cluster_v = []
		for yolo in range(0,len(selected_features)):
			selected_data = np.array(cluster1[c])
			v = np.var(selected_data[:,yolo], dtype=np.float64)
			cluster_v.append(v)
		variances.append(cluster_v)

	headers = np.array(headers)
	indices = np.array(selected_features)
	print("____________ VARIANCE OF EACH DIMENSION __________\n")
	print(pd.DataFrame.from_items([('Cluster A', variances[0]), ('Cluster B', variances[1])],orient='index', columns=headers[indices]))

def train_nn(training_data):
	x_train = Training_bag_of_words_features
	y_train = training_data[:, -1].astype('float')
	model = Sequential()
	model.add(Dense(10, input_dim=len(x_train[0]), activation='relu'))
	model.add(Dropout(0.45))
	model.add(Dense(1, activation='sigmoid'))
	sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mean_squared_error',
				  optimizer=sgd,
				  metrics=['accuracy'])
	model.fit(x_train, y_train, epochs=40, validation_split = 0.1)
	return model

def load_word_embeddings():
	(embeddings, headers) = read_lines_from_file('data/raop_embeddings.csv')
	wordvec_map = {}
	for row in embeddings:
		wordvec_map[row[0]] = np.array(row[1:]).astype('float')
	Word2VecModel = wordvec_map
	# vectorizer = TfidfEmbeddingVectorizer(Word2VecModel)

class NumericFeaturesExtractor(BaseEstimator, TransformerMixin):

	def fit(self, x, y=None):
		return self

	def transform(self, training_data):
		return training_data[:, 1:24].astype('float')

class BagOfWordsExtractor(BaseEstimator, TransformerMixin):

	def __init__(self):
		self.vectorizer = CountVectorizer(analyzer = "word", tokenizer = nltk.word_tokenize, preprocessor = None, stop_words = stopwords.words('english'), max_features = 5000, lowercase = True, ngram_range = (1,2))

	def fit(self, data, y=None):
		self.vectorizer.fit([x[-3] +" "+x[-2] for x in data])
		return self

	def transform(self, data):
		return self.vectorizer.transform([x[-3] +" "+x[-2] for x in data])

class Word2VecExtractor(BaseEstimator, TransformerMixin):

	def __init__(self):
		self.vectorizer = TfidfEmbeddingVectorizer(Word2VecModel)

	def fit(self, data, y=None):
		self.vectorizer.fit([x[-3] +" "+x[-2] for x in data])
		return self

	def transform(self, data):
		return self.vectorizer.transform([x[-3] +" "+x[-2] for x in data])

def run_on_feature_union():
	load_word_embeddings()
	(training_data, _) = read_lines_from_file('data/filtered_features.csv')
	training_data = np.array(training_data)

	#training_data = generate_normalized_data(training_data)
	clf = RandomForestClassifier(n_estimators=100, class_weight = "balanced")
	clf.classes_ = [0, 1]
	clf1 = LogisticRegression(penalty = 'l1', class_weight='balanced')
	clf1.classes_ = [0, 1]
	adaboost = AdaBoostClassifier(n_estimators=100)
	adaboost.classes_ = [0, 1]
	svm_clf = svm.SVC(probability = True)
	svm_clf.classes_ = [0, 1]
	randomized = ExtraTreesClassifier(n_estimators=45, max_depth=None, min_samples_split=2, class_weight = "balanced")
	randomized.classes_ = [0, 1]
	pipeline = Pipeline([('features', FeatureUnion([
		('numeric_features', NumericFeaturesExtractor()),
		('bag_of_words_features', BagOfWordsExtractor()),
		('w2v_features', Word2VecExtractor())
		], transformer_weights={
            'numeric_features': 0.8,
            'bag_of_words_features': 0.5,
            'w2v_features': 1.0,
        })), ('clf', clf)])
	pipeline.fit(training_data, training_data[:, -1].astype('float'))
	scores = cross_val_score(pipeline, training_data, training_data[:, -1].astype('float'), cv = 10)
	print(scores)
	print(np.mean(scores))

if __name__ == '__main__':
	generate_feature_files()
	print("YOLO done generating features")
	(training_data, _) = read_lines_from_file('data/filtered_features.csv')
	(test_data, _) = read_lines_from_file('data/test_feature_file.csv')
	test_data = np.array(test_data)
	training_data = np.array(training_data)

	generate_gaussian_mixture_models(training_data,test_data,"EM_numeric-features-prediction.csv")

	#training_data = generate_normalized_data(training_data)
	
	regression = train_Logistic_regression(training_data)
	
	rf_est = [50, 75, 100, 125, 150]
	dt_depth = [10, 20, 30, 40, 50]
	erf_est = [5, 10, 15, 20, 25]
	ada_est = [50, 75, 100, 125, 150]

	gnb = train_gaussian_NB(training_data)
	regression = train_Logistic_regression(training_data)

	print(regression.get_params())


	rf_results = []
	for e in rf_est:
		forest, rf_score = train_random_forest_classifier(training_data, e)
		rf_results.append(rf_score)
	rf_optimal, _ = train_random_forest_classifier(training_data, rf_est[rf_results.index(max(rf_results))])
	
	dt_results = []
	for d in dt_depth:
		dtree, dt_score = train_decision_tree_classifer(training_data, d)
		dt_results.append(dt_score)
	dt_optimal, _ = train_decision_tree_classifer(training_data, dt_depth[dt_results.index(max(dt_results))])

	erf_results = []
	for e in erf_est:
		extra_random, erf_score = train_extra_Randomized_forest_classifer(training_data, e)
		erf_results.append(erf_score)
	erf_optimal, _ = train_extra_Randomized_forest_classifer(training_data, erf_est[erf_results.index(max(erf_results))])

	ada_results = []
	for a in ada_est:
		adaboost, ada_score = train_AdaBoost_classifier(training_data, a)
		ada_results.append(ada_score)
	ada_optimal, _ = train_AdaBoost_classifier(training_data, ada_est[ada_results.index(max(ada_results))])


	#classifier = train_ensemble_classifier(training_data, rf_optimal, dt_optimal, ada_optimal, erf_optimal, gnb, regression)


	rf_optimal,_ = train_random_forest_classifier(training_data)
	generate_submission_file(rf_optimal, "numeric-features-prediction.csv")

	ensemble_classifier = train_ensemble_classifier(training_data, rf_optimal, dt_optimal, ada_optimal, erf_optimal, gnb, regression)

	filename = "ensemble-prediction.csv"
	generate_submission_file(ensemble_classifier, filename)
	
	filename = "randomized-forest-prediction.csv"
	generate_submission_file(rf_optimal, filename)

	filename = "adaboost-prediction.csv"
	generate_submission_file(ada_optimal, filename)	
	
	print('Done!')
	
	# run_on_feature_union()
