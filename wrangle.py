import pandas as pd
import env
import os
from scipy import stats
import sklearn.preprocessing
from sklearn.model_selection import train_test_split

def get_connection(db_name, username = env.username, host=env.host, password=env.password):
    '''
    This function makes a connection with and pulls from the CodeUp database. It 
    takes the database name as its argument, pulls other login info from env.py.
    Make sure you save this as a variable or it will print out your sensitive user
    info as plain text. 
    '''
    return f'mysql+pymysql://{username}:{password}@{host}/{db_name}'
    
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

def wrangle_zillow(db_name = 'zillow', username = env.username, password = env.password, host = env.host):
    ''' 
    Checks for zillow.csv file and imports it if present. If absent, it will pull in bedroom bathroom counts, sq ft.
    tax value dollar count, year built, tax amount, and fips from properties 2017 in the zillow database. Then it will drop
    nulls and drop duplicates'''
    filename = 'zillow.csv'
    if os.path.isfile(filename):
        zillow_df = pd.read_csv(filename, index_col=0)
        return zillow_df
    else:
        zillow_df = pd.read_sql('''SELECT bedroomcnt AS bed, 
                                          bathroomcnt AS bath, 
                                          calculatedfinishedsquarefeet AS sqft, 
                                          taxvaluedollarcnt AS tax_value, 
                                          yearbuilt, 
                                          taxamount, 
                                          fips 
                                    FROM properties_2017
                                    JOIN predictions_2017 USING(parcelid)
                                    WHERE propertylandusetypeid = 261 
                                    AND transactiondate BETWEEN '2017-05-01' AND '2017-08-31';''', get_connection('zillow'))
        zillow_df = zillow_df.dropna()
        zillow_df = zillow_df.drop_duplicates()
        zillow_df.to_csv('zillow.csv')
        return zillow_df 

def train_validate_test_split(df, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed)
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed)
    return train, validate, test
    
def minmax_scale(data_set, X_train):
    '''
    minmax_scale(data_set, X_train)
    Takes in the dataframe and applies a minmax scaler to it. Can pass a dataframe slice, 
    needs to be numbers. Outputs a scaled dataframe.  
    '''
    scaler = sklearn.preprocessing.MinMaxScaler().fit(X_train)
    x_scaled = scaler.transform(data_set)
    x_scaled = pd.DataFrame(x_scaled)
    x_scaled.columns = data_set.columns
    return x_scaled

    scaler = MinMaxScaler(copy=True).fit(X_train[numeric_cols])

    # scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train.
    #
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

def std_scale(data_set):
    '''
    Takes in the dataframe and applies a standard scaler to it. Can pass a dataframe slice, 
    needs to be numbers. Outputs a scaled dataframe.  
    '''
    scaler = sklearn.preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(data_set)
    x_scaled = pd.DataFrame(x_scaled)
    x_scaled.columns = data_set.columns
    return x_scaled

def robust_scale(data_set):
    '''
    Takes in the dataframe and applies a robust scaler to it. Can pass a dataframe slice, 
    needs to be numbers. Outputs a scaled dataframe.  
    '''
    scaler = sklearn.preprocessing.RobustScaler()
    x_scaled = scaler.fit_transform(data_set)
    x_scaled = pd.DataFrame(x_scaled)
    x_scaled.columns = data_set.columns
    return x_scaled

def quant_transformer(data_set, output_dist = 'normal'):
    """
    Takes in a dataframe and applies a quantile transormer to it. Defau
    Returns a transformed dataframe with renamed columns. Defalt distribution is normal, but can pass uniform for a uniform distribution"""
    qt = sklearn.preprocessing.QuantileTransformer(output_distribution = output_dist)
    x_scaled = qt.fit_transform(data_set)
    x_scaled = pd.DataFrame(x_scaled)
    x_scaled.columns = data_set.columns
    return x_scaled

def months_to_years(data_set):
    data_set['tenure_years'] = round(data_set.tenure / 12, 0)
    data_set = data_set.rename(columns={'tenure': 'tenure_month'})
    return data_set

def get_dummies(df, object_cols):
    """
    This function takes in a dataframe and list of object column names,
    and creates dummy variables of each of those columns.
    It then appends the dummy variables to the original dataframe.
    It returns the original df with the appended dummy variables.
    """

    # run pd.get_dummies() to create dummy vars for the object columns.
    # we will drop the column representing the first unique value of each variable
    # we will opt to not create na columns for each variable with missing values
    # (all missing values have been removed.)
    dummy_df = pd.get_dummies(object_cols, dummy_na=False, drop_first=True)

    # concatenate the dataframe with dummies to our original dataframe
    # via column (axis=1)
    df = pd.concat([df, dummy_df], axis=1)

    return df

def encode_values(df):
    '''
    Takes a dataframe and returns a new dataframe with encoded categorical variables
    '''
    label_encoder = sklearn.preprocessing.LabelEncoder()
    for x in df.select_dtypes(include = 'category'):
        df[x] = label_encoder.fit_transform(df[x])
    return df