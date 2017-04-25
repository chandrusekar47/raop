import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def read(filename):
	with open(filename, 'r') as f:
		data = list(csv.reader(f))
	labels = data.pop(0)
	return data, labels

def count(predicate, sequence):
    return sum(1 for s in sequence if predicate(s))

# Globals / constants
NUM_NARRATIVES = 5
narratives = ['money', 'job', 'student', 'family', 'craving']
triggers = [
	['money', 'now', 'broke', 'week', 'until', 'time', 'last', 'day', 'when', 'today', 'tonight', 'paid', 'next', 'first', 'night', 'after', 'tomorrow', 'month', 'while', 'account', 'before', 'long', 'Friday', 'rent', 'buy', 'bank', 'still', 'bills', 'bills', 'ago', 'cash', 'due', 'due', 'soon', 'past', 'never', 'paycheck', 'check', 'spent', 'years', 'poor', 'till', 'yesterday', 'morning', 'dollars', 'financial', 'hour', 'bill', 'evening', 'credit', 'budget', 'loan', 'bucks', 'deposit', 'dollar', 'current', 'payed'],
	['work', 'job', 'paycheck', 'unemployment', 'interview', 'fired', 'employment', 'hired', 'hire'],
	['college', 'student', 'school', 'roommate', 'studying', 'university', 'finals', 'semester', 'class', 'study', 'project', 'dorm', 'tuition'],
	['family', 'mom', 'wife', 'parents', 'mother', 'husband', 'dad', 'son', 'daughter', 'father', 'parent', 'mum'],
	['friend', 'girlfriend', 'craving', 'birthday', 'boyfriend', 'celebrate', 'party', 'game', 'games', 'movie', 'date', 'drunk', 'beer', 'celebrating', 'invited', 'drinks', 'crave', 'wasted', 'invite']
]

# Read data and find necessary indices
data, labels = read('data/filtered_features_1.csv')
requestIdx = labels.index('edited_text')
titleIdx = labels.index('title')
successIdx = labels.index('success')

# Get narratives for each record
results = []
for d in data:
	text = d[titleIdx] + ' ' + d[requestIdx]
	counts = [0, 0, 0, 0, 0]
	for w in text.split(' '):
		for t in range(NUM_NARRATIVES):
			if w in triggers[t]:
				counts[t] += 1

	narr = narratives[counts.index(max(counts))]
	results.append([narr, int(d[successIdx])])

# Get count of success and failure for each narrative and normalize
metrics = [[n, count(lambda r: r[0] == n and r[1] == 1, results), count(lambda r: r[0] == n and r[1] == 0, results)] for n in narratives]
for m in metrics:
	total = m[1] + m[2]
	m[1] /= float(total)
	m[2] /= float(total)

# Plot results
fig, ax = plt.subplots()
width = 0.35
ind = np.arange(NUM_NARRATIVES)
succ = ax.bar(ind, [m[1] for m in metrics], width, color='g')
fail = ax.bar(ind + width, [m[2] for m in metrics], width, color='r')
ax.set_ylabel('Percentage')
ax.set_title('Success & Failure of Various Narratives')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(narratives)
ax.legend((succ[0], fail[0]), ('Success', 'Failure'))
plt.show()
