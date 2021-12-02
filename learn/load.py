'''load.py Load loads the dataset, cleans it, and generates a new clean
csv file.
'''

#standard imports
import pandas as pd


def load(filename):
  '''load(filename) takes as input a filename and loads a dataframe.
  '''
  return pd.read_csv(filename) #demo


def clean(df):
  '''clean(df) takes as input a dataframe and fixes any data errors  
     *that might affect your results*. it returns a copy of the dataframe 
     without errors.
  '''
  
  return df #demo


#The main() function  of this program

if __name__ == "__main__":
    df = load('data_2.csv')
    cdf = clean(df)
    cdf.to_csv('clean.csv')
