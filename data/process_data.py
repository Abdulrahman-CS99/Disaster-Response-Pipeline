import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3



def load_data(messages_filepath, categories_filepath):
    """
    Function: loading two data sests from csv files and merging them

    Args:
        messages_filepath (str): messages file path
        categories_filepath (str): categories file path
        
    Return:
    df: merge dataframe from messages and categories  
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    
    return df


def clean_data(df):
    """ Function: cleaning the dataset and return it after the cleaning

    Args:
        df (pandas dataframe): raw dataset
        
        Return:
        df (pandas dataframe): the cleaned dataset
    """# create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0] 
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.split('-')[0]).values.tolist()
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
     categories[column] = categories[column].apply(lambda x: x.split('-')[1]) 
     categories[column] = categories[column].astype(int)
    
    arr=[]
    for i in range(categories.shape[1]):
        for j in range(categories.shape[0]):
            if (categories.iloc[j][i] !=0 and categories.iloc[j][i] != 1):
                arr.append(j)
    categories.drop((arr[i] for i in range(len(arr))),inplace=True)

    # drop the original categories column from `df`
    df.drop(columns={'categories'},inplace=True)
    df = pd.concat([df,categories], axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    """Function : Saving clean data into a databse

    Args:
        df (pandas df): cleaned data
        database_filename (str): name of the database
    """
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql("messages", engine,if_exists='replace', index=False)


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