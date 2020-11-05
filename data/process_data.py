import sys
import nltk #for natural language processing 
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd

#nltk pakages 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#sklearn pakages

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages =  pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on=('id'))
    return df
  
def clean_data(df):
    """ 
    Data Cleaning Process. 
  
    Takes In A Pandas DataFrame And Seperates The Categories Into Individual
    Category Columns And Then Merging Them Back To The DataFrame ,Finally 
    Returning New DataFrame. 
  
    Parameters: 
    df (DataFrame): Pandas DataFrame Object Containing The Data.
  
    Returns: 
    cleaned_df (DataFrame): Pandas DataFrame Object Containing The Cleaned Data.
  
    """

    # creating a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand = True)

    # selecting the first row of the categories dataframe to extract column names
    row = categories.loc[0]

    # creating a list of category column names
    category_colnames = [category[:len(category)-2] for category in row ]

    # renaming the columns of `categories` dataframe
    categories.columns = category_colnames

    # now converting category values to just numbers 0 or 1
    for column in categories:

    	# setting each value to be the last character of the string
    	categories[column] = categories[column].str[-1]
    
    	# converting column from string to numeric
    	categories[column] = pd.to_numeric(categories[column])

    # replacing categories column in df with new category columns.
    df = df.drop('categories',axis = 1)

    # concatenating the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis = 1)

    # removing duplicates
    cleaned_df = df.drop_duplicates(keep = 'first')
    
    return cleaned_df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('DisasterResponse', engine, index=False)
    pass

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()