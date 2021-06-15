import pandas as pd
import env
import os
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
    Imports single residential family properties from the zillow database. columns are bedroom/bathroom counts,
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