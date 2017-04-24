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
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier




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

def dict_to_csv(filename, output_file_name):
	with open(filename) as data_file:
		data = json.load(data_file)	
	output_file = open(output_file_name, 'w')
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
	"subreddits_posted",
	"title",
	"edited_text",
	"success"]
	sid = SentimentIntensityAnalyzer()
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
		scores = sid.polarity_scores(post_text)
		values.append(scores['pos'])
		values.append(scores['neg'])
		values.append(scores['neu'])
		values.append('"'+array_to_str(record["requester_subreddits_at_request"]) +'"')
		values.append('"'+cleanup(record["request_title"])+ '"')
		values.append('"'+post_text + '"')
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

def add_sentiment_scores(lines_in_file, headers):
	text_index = 15
	headers.insert(13, "pos_score")
	headers.insert(14, "neg_score")
	headers.insert(15, "neutral_score")
	sid = SentimentIntensityAnalyzer()
	for line in lines_in_file:
		post_text = line[text_index]
		line[text_index] = line[text_index].replace("\n", "")
		line[text_index] = line[text_index].replace("\t", "")
		line[text_index] = line[text_index].replace("\r", "")
		scores = sid.polarity_scores(post_text)
		line.insert(13, scores['pos'])
		line.insert(14, scores['neg'])
		line.insert(15, scores['neu'])

def generate_bag_of_word_features(post_texts):
	vectorizer = CountVectorizer(analyzer = "word", tokenizer = nltk.word_tokenize, preprocessor = None, stop_words = stopwords.words('english'), max_features = 10000)
	return vectorizer.fit_transform(post_texts).toarray()

def train_random_forest_classifier():
	(training_data, _) = read_lines_from_file('data/filtered_features.csv')
	training_data = np.array(training_data)
	forest = RandomForestClassifier(n_estimators = 100)
	forest.classes_ = [0, 1]
	forest = forest.fit(training_data[:, 1:15].astype('float'), training_data[:, -1].astype('float'))
	# forest.fit_transform(generate_bag_of_word_features(training_data[:, -2]), training_data[:, -1].astype('float'))	
	# scores = cross_val_score(forest, generate_bag_of_word_features(training_data[:, -2]), training_data[:, -1].astype('float'), cv = 5)
	scores = cross_val_score(forest, training_data[:, 1:15].astype('float'), training_data[:, -1].astype('float'), cv = 5)
	print(scores)
	print(np.mean(scores))
	return forest

def generate_submission_file(classifier, submission_filename):
	(test_data, _) = read_lines_from_file('data/test_feature_file.csv')
	test_data = np.array(test_data)
	output_probabs=classifier.predict_proba(test_data[:, 1:15].astype('float'))
	output_file = open(submission_filename, 'w')
	print("request_id,requester_received_pizza", file = output_file)
	for ind, record in enumerate(test_data):
		print("%s,%f"%(record[0], output_probabs[ind][1]), file = output_file)
	output_file.close()

def generate_feature_files():
	dict_to_csv('data/train.json', 'data/filtered_features.csv')
	dict_to_csv('data/test.json', 'data/test_feature_file.csv')


def voting_classifier():
	(training_data, _) = read_lines_from_file('data/filtered_features.csv')
	training_data = np.array(training_data)
	clf1 = LogisticRegression(random_state=1)
	clf2 = RandomForestClassifier(random_state=1)
	clf3 = GaussianNB()
	X = (training_data[:, 1:15].astype('float'))
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
	#print(eclf3.predict_proba(test_data[:, 1:15].astype('float')))
	scores = cross_val_score(eclf3, training_data[:, 1:15].astype('float'), training_data[:, -1].astype('float'), cv = 5)
	print(scores)
	print(np.mean(scores))
	return eclf3


if __name__ == '__main__':
	generate_feature_files()
	classifier=voting_classifier()
	#classifier = train_random_forest_classifier()
	generate_submission_file(classifier, "numeric-features-prediction.csv")