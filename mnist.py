# Script used to create Mnist_mini and Mnist_full datasets.

import numpy as np
from sklearn.datasets import fetch_mldata
from pandas import DataFrame

# Default download location for caching is 
# ~/scikit_learn_data/mldata/mnist-original.mat unless specified otherwise.
mnist = fetch_mldata('MNIST original')

# Create DataFrame, group data by class.
df = DataFrame(mnist.data)
df['class'] = mnist.target
grouped = df.groupby('class')

# Write data feature values to file in Dataset directory by class.
for name, group in grouped:
	# Create mini binary MNIST classification dataset for faster testing.
	if int(name) < 2:
		fname = 'Dataset/Mnist_mini/Class' + str(int(name)) + '.txt'
		np.savetxt(fname=fname, X=group[:200], fmt='%d',delimiter='\t',newline='\n')

	# Create full MNIST classification for full application.
	fname = 'Dataset/Mnist_full/Class' + str(int(name)) + '.txt'
	np.savetxt(fname=fname, X=group, fmt='%d', delimiter='\t', newline='\n')