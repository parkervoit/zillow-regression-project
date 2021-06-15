
import pandas as pd
import env
import os
from scipy import stats
import sklearn.preprocessing
from sklearn.model_selection import train_test_split

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

def create_dummies(df, object_cols):
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