from __future__ import print_function
import json
import re
import csv
import numpy as np
import nltk
import datetime
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
import sklearn.preprocessing

sentiment_analyzer = SentimentIntensityAnalyzer()
vectorizer = CountVectorizer(analyzer = "word", tokenizer = nltk.word_tokenize, preprocessor = None, stop_words = stopwords.words('english'), max_features = 10000)
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
selected_features = [1,2,4,6,7,8,14,16,17,18,27]

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

	print("LOG REGRESSION COEFF : "+str(clf1.coef_) )#acess coefficients
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
	print(training_data[0])
	return training_data

def voting_classifier():
	(training_data, _) = read_lines_from_file('data/filtered_features.csv')
	training_data = np.array(training_data)
	clf1 = LogisticRegression(random_state=1)
	clf2 = RandomForestClassifier(random_state=1)
	clf3 = GaussianNB()
	X = (training_data[:, selected_features].astype('float'))
	y = (training_data[:, -1].astype('float'))
	eclf1 = VotingClassifier(estimators=[
	        ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
	eclf1 = eclf1.fit(X, y)
	print(eclf1.predict(X))

	eclf2 = VotingClassifier(estimators=[
	        ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
	        voting='soft')
	eclf2 = eclf2.fit(X, y)
	print(eclf2.predict(X))

	eclf3 = VotingClassifier(estimators=[
	       ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
	       voting='soft', weights=[2,1,1])
	eclf3 = eclf3.fit(X, y)
	#print(eclf3.predict_proba(test_data[:, selected_features].astype('float')))
	scores = cross_val_score(eclf3, training_data[:, selected_features].astype('float'), training_data[:, -1].astype('float'), cv = 5)
	print(scores)
	print(np.mean(scores))
	return eclf3


if __name__ == '__main__':
	#generate_feature_files()
	#print("YOLO done generating features")
	(training_data, _) = read_lines_from_file('data/filtered_features.csv')
	training_data = np.array(training_data)

	#training_data = generate_normalized_data(training_data)
	#print("Normalization done")
	#print(training_data[0])

	regression = train_Logistic_regression(training_data)

	print(regression.get_params())

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


	ensemble_classifier = train_ensemble_classifier(training_data, rf_optimal, dt_optimal, ada_optimal, erf_optimal, gnb, regression)

	filename = "ensemble-prediction.csv"
	generate_submission_file(ensemble_classifier, filename)
	
	filename = "randomized-forest-prediction.csv"
	generate_submission_file(rf_optimal, filename)

	filename = "adaboost-prediction.csv"
	generate_submission_file(ada_optimal, filename)	

	print('Done!')
