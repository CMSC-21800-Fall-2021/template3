'''train.py loads a clean csv and trains a model on it.
'''

#standard imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def train(X_train, Y_train):
	'''Takes in X_train and Y_train should output a model
	'''
	return None


def evaluate(model, X_test, Y_test):
	'''Takes in X_test and Y_test should output a classification report
	'''
	return None


def featurize(df):
	'''
	   X - Should take a dataframe of feature columns and
	   output a 2D NxK numpy of N data points and K dimensional
	   features. 
	'''
	return df.to_numpy()


#The main() function  of this program
if __name__ == "__main__":
	cdf = pd.read_csv('clean.csv')

    #Q2.TODO 
	label = 'DAMAGE'
	features =  []#

	X = featurize(cdf[features])
	Y = cdf.to_numpy()


	X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

    #Q3.TODO
	model = train(X_train,Y_train)

	#Q4.TODO
	print(evaluate(model, X_test, Y_test))





