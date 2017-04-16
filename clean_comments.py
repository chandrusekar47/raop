import csv

def read(fileName):
	with open(fileName, 'r') as f:
		data = list(csv.reader(f))
	labels = data.pop(0)
	return data, labels

def write(fileName, labels, data):
	with open(fileName, 'w') as f:
		writer = csv.writer(f, delimiter=',')
		writer.writerow(labels)
		writer.writerows(data)

data, labels = read('data/comments_cleaned.csv')

labels[0] = 'id'

clean = []
for d in data:
	c = []
	for s in d:
		c.append(s.replace('\n', ' ').replace('\r', ' '))
	clean.append(c)

write('data/comments_cleaned2.csv', labels, clean)