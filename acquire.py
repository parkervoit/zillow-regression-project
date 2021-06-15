import pandas as pd
import env
import os
from scipy import stats
# must have env.py saved in same directory as script. ensure the env.py is in your .gitignore
def get_connection(db_name, username = env.username, host=env.host, password=env.password):
    '''
    This function makes a connection with and pulls from the CodeUp database. It 
    takes the database name as its argument, pulls other login info from env.py.
    Make sure you save this as a variable or it will print out your sensitive user
    info as plain text. 
    '''
    return f'mysql+pymysql://{username}:{password}@{host}/{db_name}'
    
def get_telco_db(db_name = 'telco_churn', username = env.username, password = env.password, host = env.host):
    filename = 'telco.csv'
    if os.path.isfile(filename):
        telco_df = pd.read_csv(filename, index_col=0)
        return telco_df
    else:
        telco_df = pd.read_sql('''SELECT * FROM customers 
                          JOIN internet_service_types USING(internet_service_type_id)
                          JOIN contract_types USING(contract_type_id)
                          JOIN payment_types USING (payment_type_id);''',
                        get_connection('telco_churn'))
        telco_df.to_csv(filename)
        return telco_df
    
def get_zillow_db(db_name = 'zillow', username = env.username, password = env.password, host = env.host):
    '''
    Imports single residential family properties from the zillow database. columns are parcelid, bedroom/bathroom counts,
    square footage, tax value, year it was built, tax, and fips for the year 2017'''
    filename = 'zillow.csv'
    if os.path.isfile(filename):
        zillow_df = pd.read_csv(filename, index_col=0)
        return zillow_df
    else:
        zillow_df = pd.read_sql('''SELECT parcelid,
                                          bedroomcnt AS bed, 
                                          bathroomcnt AS bath, 
                                          calculatedfinishedsquarefeet AS sqft, 
                                          taxvaluedollarcnt AS tax_value, 
                                          yearbuilt, 
                                          taxamount, 
                                          fips 
                                    FROM properties_2017
                                    JOIN predictions_2017 USING(parcelid)
                                    WHERE propertylandusetypeid = 261 
                                    AND transactiondate BETWEEN '2017-05-01' AND '2017-08-31';''',
                        get_connection('zillow'))
        zillow_df.to_csv(filename)
        return zillow_df
    
def missing_values_table(df):
    '''
    this function takes a dataframe as input and will output metrics for missing values, and the percent of that column that has missing values
    '''
        # Total missing values
    mis_val = df.isnull().sum()
        # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
        # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
    mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
        # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
           "There are " + str(mis_val_table_ren_columns.shape[0]) +
           "columns that have missing values.")
        # Return the dataframe with missing information
    return mis_val_table_ren_columns