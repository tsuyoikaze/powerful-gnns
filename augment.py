import sys
import os
from numpy.random import randint
import pandas as pd

# Usage: python augment.py <dataset folder> <offset_low> <offset_high (inclusive)>

def augment(data, offset):
    return data + randint(offset[0], offset[1] + 1, size = data.shape)

dir_list = []

for root, dirs, files in os.walk(sys.argv[1]):
    for file in files:
        dir_list.append((root, file))

for root, file in dir_list:
    csv = pd.read_csv(os.path.join(root, file))
    if 'graph' in file:
        offset = (float(sys.argv[2]), float(sys.argv[3]))
        csv = augment(csv, offset)
    csv.to_csv(os.path.join(root, 'augmented_' + file))
