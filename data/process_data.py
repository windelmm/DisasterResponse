import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    merged_df = messages.merge(categories, on= 'id', how= 'left')
    
    return merged_df
                         
def clean_data(merged_df):
    catDf = merged_df['categories'].str.split(';', expand= True)
    
    firstRow = catDf.loc[0, :]
    
    cat_colnames = firstRow.apply(lambda x: x.split('-')[0]).values.tolist()
    
    catDf.columns = cat_colnames
    
    for column in catDf:
        catDf[column] = catDf[column].apply(lambda x: x.split('-')[1])
        catDf[column] = catDf[column].astype(int)
        catDf.loc[catDf[column] > 1, column] = 1
        
    merged_df.drop('categories', axis=1, inplace=True)

    df = pd.concat([merged_df, catDf], axis=1)

    df.drop_duplicates(inplace= True)

    return df

def save_data(df, database_filename):
    name = 'sqlite:///' + database_filename  
    engine = create_engine(name)
    
    df.to_sql('DisasterResponse', engine, index= False)

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