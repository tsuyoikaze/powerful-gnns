import sys
import os
import numpy as np 
import json
from subprocess import call

# usage: python split_validation.py <path of training set> <path of validation set> <ratio of data removed from training> <patient_to_labels.json>

l = os.listdir(sys.argv[1])
os.mkdir(sys.argv[2])
d = dict()
patient_to_labels = json.load(open(sys.argv[4]))
for i in l:
	if patient_to_labels[i] not in d:
		d[patient_to_labels[i]] = []
	d[patient_to_labels[i]].append(i)

res = []

for i in d:
	num = int(len(d[i]) * float(sys.argv[3]))
	if num < 1:
		print('In class %d, epected number of patient in validation is %f' % (i, len(d[i]) * float(sys.argv[3])))
	idx = np.random.choice(int(len(d[i])), size=num)
	for patient in idx:
		print(patient)
		res.append(patient)

for i in res:
	call('cp -r %s %s' % (os.path.join(sys.argv[1], str(i)), os.path.join(sys.argv[2], str(i))))