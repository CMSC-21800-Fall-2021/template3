'''scratch.py

You will write down any "scratch" code that you use to explore the dataset in
this file. This code does not produce any outputs (it does not modify any files)
but allows you to explore the data. We're separating this code from the rest because 
it allows us to understand your thought process during data exploration.

*Remember to replace the return statements with your code*
'''

#standard imports
import pandas as pd
import matplotlib.pyplot as plt


#TODO 1.
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


#TODO 2.
def plot(df):
  '''plot(df) takes as input a clean dataframe and generates a plot. The 
  style of the plot and variables are up to you.
  '''
  
  plt.scatter(df['A'],df['B'])
  return None



#The main() function  of this program

if __name__ == "__main__":
    df = load('data.csv')
    cdf = clean(df)
    plot(cdf)
    plt.show()


