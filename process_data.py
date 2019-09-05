# import libraries
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    print(messages.head())
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    print(categories.head())
    # drop original as these are in an other language that you can't add much value
    messages.drop(columns=['original'],axis=1,inplace=True)
    messages[messages.duplicated(['id','message','genre'])].count()
    #### There are 68 ids that are duplicated in the messages dataframe. Drop them
    messages.drop_duplicates(inplace=True)
    messages.info()
    
    categories[categories.duplicated(['id','categories'])].count()
    #### There are ~32 duplicates in the categories data frame. Drop them
    categories.drop_duplicates(inplace=True)
    categories.info()
    
    categories[categories.duplicated(['id'])].count()
    # There are 36 ids that are duplicated but the categories are not hence these are still in the Data Frame
    # There are still some records in categories where ids are duplicated but there's something else in the categories column
    
    # Split the values in the categories column on the ; character so that each value becomes a separate column. You'll find this method very helpful! Make sure to set expand=True. 
    # Use the first row of categories dataframe to create column names for the categories data. Rename columns of categories with new column names.
    
    # create a dataframe of the 36 individual category columns
    categories_expanded = categories.categories.str.split(';',None,True)
    categories_expanded.head()
    categories.head()
    
    categories_full = pd.concat([categories, categories_expanded],axis=1)
    categories_full.drop(columns=['categories'],axis=1,inplace=True)
    categories_full.head()
    
    names = []
    for x in categories_full.iloc[0]:
        names.append(x)
        
    #column_names
    names = names[1:]
    columns = [name[:-2] for name in names]
    columns    
    columns = ['id'] + columns
    columns
    
    categories_full.columns = columns
    categories_full.head()
    
    categories_full.set_index(keys='id',inplace=True)
    categories_full.head()
    
    categories_full.columns[1:]
    
    for column in categories_full.columns:
    # set each value to be the last character of the string
        categories_full[column] = categories_full[column].str[-1]
    
    # convert column from string to numeric
        categories_full[column] = categories_full[column].astype(int)
    categories_full.head()
    
    categories_full.describe(include='all')
    
    # There are some rows for which the related column has value of 2 - that seems like a data quality issue
    # Also, the child_alone column has only 0
    categories_full.query('related=="2"')['related'].count()
    
    categories_full['related'].replace(\
                                   to_replace=2,\
                                   value=1,\
                                   inplace=True\
                                  )
    categories_full.describe(include='all')
    
    categories_full[categories_full.duplicated()].count()
    
    categories_updated = categories_full.groupby(['id'])['related', 'request', 'offer', 'aid_related', \
                                          'medical_help','medical_products', 'search_and_rescue', \
                                          'security', 'military','child_alone', 'water', 'food', \
                                          'shelter', 'clothing', 'money','missing_people', 'refugees', \
                                          'death', 'other_aid','infrastructure_related', 'transport', \
                                          'buildings', 'electricity','tools', 'hospitals', 'shops', \
                                          'aid_centers', 'other_infrastructure','weather_related', \
                                          'floods', 'storm', 'fire', 'earthquake', 'cold',\
                                          'other_weather', 'direct_report'].agg('max').reset_index()
    categories_updated.head()
    categories_updated.info()
    
    # Remove child_alone column as all values are 0 and there is no variation
    categories_updated.drop(columns=['child_alone'],axis=1,inplace=True)
    categories_updated.info()
    
    # Merge the DataFrames
    # There are same number of records in messages and categories
    
    # merge datasets
    df = messages.merge(categories_updated, how='left',on='id')
    df.head()
    print('Data Loaded')
    return df






def clean_data(df):
    print('Cleaning begins')
    df.describe(include='all')
    # #NAME? seems to be a message thats been duplicated
    # these 4 messages cant be interpreted in any way and so its best to remove them
    
    df.drop(df.index[df['message'] == "#NAME?"], inplace = True)
    df.describe(include='all')
    
    corr = df.drop(columns=['id','message','genre']).corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
          
    
    print('going to print matplotlib')
    # Set up the matplotlib figure
    #f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    #cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    #sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
    for column in df.columns[3:]:
        print("---{}---".format(column))
        print(df[column].value_counts())
        
    # check number of duplicates
    df[df.duplicated()].count()    
    return df



def save_data(df, database_filename):
    qstring = 'sqlite:///'+database_filename
    engine = create_engine(qstring)
    df.to_sql('messages', engine, index=False, if_exists='replace')
    
    return  


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