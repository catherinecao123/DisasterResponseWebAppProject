import sys
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# import warnings
# warnings.filterwarnings("ignore")

def load_data(messages_filepath, categories_filepath):
    """
    read in files messages and categories and merge the two files to created data frame df 
    INPUT: messages filepath, categories filepath
    OUTPUT: data frame df
    """
    messages = pd.read_csv(messages_filepath,dtype=str)
    categories = pd.read_csv(categories_filepath,dtype=str)
    df = messages.merge(categories,how='inner',on='id')
    return df

def clean_data(df):
    """
    INPUT: 
    OUTPUT:
    """

    
    categories = df['categories'].str.split(';',expand=True)
    
    row = categories.iloc[0]
  
    category_colnames = list(row.apply(lambda x: x[:-2]))
  
    categories.columns = category_colnames
   
    for column in categories.columns:
        
        categories[column] = categories[column].astype(str).str.split('-').str[1]
        categories[column] = pd.to_numeric(categories[column])
        #Convert the string to a numeric value
        categories[column].loc[(categories[column]!=0)&(categories[column]!=1)]=1

    df.drop(['categories'],axis=1,inplace=True)
    
    df = pd.concat([df,categories],axis=1)
    
#     df.drop(['child_alone'],axis=1,inplace=True)
   
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    """
    INPUT:
    df:

    OUTPUT:
    None
    """
    engine = create_engine('sqlite:///'+ database_filename)
    table_name = os.path.basename(database_filename).split('.')[0]
    df.to_sql(table_name, engine, index=False, if_exists='replace')

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