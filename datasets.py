import random
import json

def housing_data_gen(m = 50, n = 500):
	gen_func = lambda sq_foot: (500 * sq_foot - 0.05 * sq_foot ** 2 + 10000) * (20 - random.normalvariate(0, 1)) / 20 + random.normalvariate(0, 100000)
	return [[sq_foot * 10, gen_func(sq_foot * 10)] for sq_foot in range(m, n)]

def process_data():
	file_read = open('./breast-cancer-wisconsin.data', 'r')
	file_write = open('./breast-cancer-wisconsin.json', 'w+')
	line = file_read.readline()
	while line:
		file_write.write('[' + line[:len(line)-1].replace('?', '0') + ']\n')
		line = file_read.readline()
		file_read.close()
		file_write.close()

def cancer_data_gen():
	file_read = open('./breast-cancer-wisconsin.json', 'r')
	line = file_read.readline()
	data_array = []
	while line:
		data_array.append(json.loads(line)[1:])
		line = file_read.readline()
		return data_array
