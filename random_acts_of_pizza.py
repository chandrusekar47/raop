from __future__ import print_function
import json
import re
import csv
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def cleanup(string):
	string = re.sub("[^a-zA-Z\d\s.]", " ", string)
	string = re.sub("([^\d])\.([^\d])",r"\1 \2",string)
	string = re.sub("\s{2,}", " ", string)
	return string.strip(" ")

def dict_to_csv():
	with open('data/train.json') as data_file:
		data = json.load(data_file)
	keys = data[0].keys()
	str_fields = ['request_title', 'request_text', 'request_text_edit_aware']
	array_field = "requester_subreddits_at_request"
	other_keys = [x for x in keys if x not in str_fields and x != array_field]
	print(','.join(other_keys + [array_field] + str_fields))
	for record in data:
		normal_values = map(lambda x: str(record[x]), other_keys)
		array_value = '"'+ u' '.join(record[array_field]).encode('utf-8') +'"'
		quoted_values = map(lambda x: '"' + cleanup(record[x]) + '"', str_fields)
		print(u','.join(normal_values + [array_value] + quoted_values).encode('utf-8'))

def read_lines_from_file(input_file):
	lines = []
	with open(input_file, "rb") as file:
		reader = csv.reader(file, delimiter = ",", )
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
		scores = sid.polarity_scores(post_text)
		line.insert(13, scores['pos'])
		line.insert(14, scores['neg'])
		line.insert(15, scores['neu'])

if __name__ == '__main__':
	(lines, headers) = read_lines_from_file('data/filtered_features.csv')
	add_sentiment_scores(lines, headers)
	output_file = open("data/filtered_features_w_senti.csv", 'w')
	print(','.join(headers[1:]), file = output_file)
	for line in lines:
		print(','.join(map(str, line[1:])), file = output_file)
	output_file.close()