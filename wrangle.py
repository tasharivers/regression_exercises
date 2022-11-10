######################### Acquire Telco Function ###########################

#import libraries
import pandas as pd
import numpy as np
import os
from pydataset import data
from sklearn.model_selection import train_test_split

# acquire
import acquire


# Create helper function to get the necessary connection url.
def get_connection(db_name):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    from env import host, user, password
    return f'mysql+pymysql://{user}:{password}@{host}/{db_name}'


#create function to retrieve telco_churn data with specific columns
def get_telco_data():
    '''
    This function reads in the Telco Churn data from the Codeup db
    and returns a pandas DataFrame with customer_id, monthly_charges, tenure, total_charges columns
    for customers who have 2-year contracts.
    '''
    
    sql_query = '''
    SELECT customer_id, monthly_charges, tenure, total_charges
    FROM customers
    WHERE contract_type_id LIKE '3'
    '''
    return pd.read_sql(sql_query, get_connection('telco_churn'))

############################ ALL Telco Data Function ##############################

#create function to retrieve telco_churn data with all columns
def all_telco_data(df):
    '''
    This function reads in the Telco Churn data from the Codeup db
    and returns a pandas DataFrame with all columns
    '''
    
    sql_query = '''
    SELECT *
    FROM customers
    '''
    return pd.read_sql(sql_query, get_connection('telco_churn'))

############################ Wrangle Telco Function ##############################


def wrangle_telco():
    '''
    This function checks to see if telco_churn.csv already exists, 
    if it does not, one is created
    then the data is cleaned and the dataframe is returned
    '''
    #check to see if telco_churn.csv already exist
    if os.path.isfile('telco_churn.csv'):
        df = pd.read_csv('telco_churn.csv', index_col=0)
    
    else:

        #creates new csv if one does not already exist
        df = get_telco_data()
        df.to_csv('telco_churn.csv')

    #replace blank spaces and special characters
    df = df.replace(r'^\s*$', np.nan, regex=True)

    #change total_charges to float from object
    df.total_charges = df.total_charges.str.replace(' ', '0').astype(float)

    #fill NaN values (10 out of 1695) with monthly charges
    df.total_charges = df.total_charges.fillna(df.monthly_charges)

    return df

############################ Tenure Years Function ##############################

def months_to_years(df):
    '''
    this function accepts the telco churn dataframe
    and returns a dataframe with a new feature in complete years of tenure
    '''
    df['tenure_years'] = df.tenure / 12
    return df

###########################################################################

#################################### ZILLOW #####################################

############################ Acquire Zillow Function ##############################

def get_zillow_data():
    '''
    This function reads in the Zillow data from the Codeup db
    and returns a pandas DataFrame with cbedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, and fips
    for all Single Family Residential properties.
    '''
    
    sql_query = '''
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, and fips
    FROM properties_2017
    WHERE propertylandusetypeid = 261
    '''
    return pd.read_sql(sql_query, get_connection('zillow'))


############################ Wrangle Zillow Function ##############################
def wrangle_zillow():
    '''
    This function checks to see if zillow.csv already exists, 
    if it does not, one is created
    then the data is cleaned and the dataframe is returned
    '''
    #check to see if telco_churn.csv already exist
    if os.path.isfile('zillow.csv'):
        zillow_df = pd.read_csv('zillow.csv', index_col=0)
    
    else:

        #creates new csv if one does not already exist
        zillow_df = get_zillow_data()
        zillow_df.to_csv('zillow.csv')

    #replace blank spaces and special characters
    zillow_df = zillow_df.replace(r'^\s*$', np.nan, regex=True)

    #drop null values- at most there were 9000 nulls (this is only 0.5% of 2.1M)
    zillow_df = zillow_df.dropna()

    return zillow_df



############################ Split Data Function ##############################

def split_data(df):
    '''
    split our data,
    takes in a pandas dataframe
    returns: three pandas dataframes, train, test, and validate
    '''
    #create train_validate and test datasets
    train, test = train_test_split(df, train_size = 0.8, random_state = 123)
    #create train and validate datasets
    train, validate = train_test_split(train, train_size = 0.7, random_state = 123)

    # Have function print datasets shape
    print(f'train -> {train.shape}')
    print(f'validate -> {validate.shape}')
    print(f'test -> {test.shape}')
   
    return train, validate, test