import numpy as np 
from PIL import Image, ImageDraw
import pandas as pd 
import os
import sys
import re
from cluster import triangle_graph

pattern = r'SOB_(?P<type>B|M)_(?P<class>[A-Z]+)-(?P<id>\d+-[\d\w]+)-200-(?P<number>\d+).png'

def overlay(path, feature_path)
	
	img = Image.open(path)
	draw = ImageDraw.Draw(img)


	filename = path.split('/')[-1]
	m = re.match(pattern, filename)
	_type = m.group('type')
	_class = m.group('class')
	_id = m.group('id')
	_number = int(m.group('number'))


	for i in ['train', 'valid', 'test']:
		path = os.path.join(feature_path, i)
		contents = os.listdir(path)
		if _id in contents:
			path = os.path.join(path, 'graph_{}.csv'.format(_number))
			break

	print('found in path {}'.format(path))

	csv = pd.read_csv(i).drop(columns=['Unnamed: 0']).values


	graph = triangle_graph([path])[0]
	for e in graph.edges_iter():
		source, target = e 
		source, target = csv[source, 2:], csv[target, 2:]
		draw.line([source, target], width = 5, fill = 128)

	img.show()

if __name__ == '__main__':
	overlay(sys.argv[1], sys.argv[2])