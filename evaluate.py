import matplotlib.pyplot as plt
from scipy import stats
from math import sqrt
import numpy as np
import pandas as pd

def plot_res(actual, predicted):
    '''plot_res(actual, predicted)
    Takes in actual values, baseline values, and predicted values.
    Returns scatterplots comparing baseline and predicted residuals to actual'''
    # save residuals and create baseline
    baseline = actual.mean()
    res = predicted - actual
    base_res = baseline - actual
    #create figure and subplots
    plt.figure(figsize=(20,9))
    # actual v model
    plt.subplot(121)
    plt.hlines(0, actual.min(), actual.max(), ls=':')
    plt.scatter(actual, res)
    plt.ylim(res.min(),res.max())
    plt.ylabel('residual ($model_y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title('Actual vs Model Residual')
    # actual v baseline
    plt.subplot(122)
    plt.hlines(0, actual.min(), actual.max(), ls=':')
    plt.scatter(actual, base_res)
    plt.ylim(base_res.min(), base_res.max())
    plt.ylabel('residual ($baseline_y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title('Actual vs Baseline Residual')
    return plt.gca()

def regression_errors(actual, predicted):
    ''' regression_errors(actual, predicted)
    returns SSE, ESS, TSS, MSE, and RMSE for predicted residuals'''
    sse_res = sum((actual - predicted)  ** 2)
    ess_res = sum((predicted - actual.mean()) ** 2)
    tss_res = sse_res + ess_res
    mse_res = sse_res / len(actual)
    rmse_res = mse_res ** (1/2)
    return (print(f'|Sum of Squared Errors: {sse_res:.4}'),
            print('|-----------------------------------'),
            print(f'|Explained Sum of Squares: {ess_res:.4}'),
            print('|-----------------------------------'),
            print(f'|Total Sum of Squares : {tss_res:.4}'),
            print('|-----------------------------------'),
            print(f'|Mean Squared Error : {mse_res:.4}'),
            print('|-----------------------------------'),
            print(f'|Root Mean Squared Error : {rmse_res:.4}'))

def baseline_mean_errors(actual):
    '''baseline_mean_errors(actual)
    returns SSE, MSE, and RMSE for baseline residuals'''
    baseline = actual.mean()
    sse_base = sum((actual - baseline)  ** 2)
    mse_base = sse_base / len(actual)
    rmse_base = mse_base ** (1/2)
    return (print(f'|Sum of Squared Errors: {sse_base:.4}'),
            print('|-----------------------------------'),
            print(f'|Mean Squared Error : {mse_base:.4}'),
            print('|-----------------------------------'),
            print(f'|Root Mean Squared Error : {rmse_base:.4}'))

def baseline_median_errors(actual):
    '''baseline_median_errors(actual)
    returns SSE, MSE, and RMSE for baseline residuals'''
    baseline = actual.median()
    sse_base = sum((actual - baseline)  ** 2)
    mse_base = sse_base / len(actual)
    rmse_base = mse_base ** (1/2)
    return (print(f'|Sum of Squared Errors: {sse_base:.4}'),
            print('|-----------------------------------'),
            print(f'|Mean Squared Error : {mse_base:.4}'),
            print('|-----------------------------------'),
            print(f'|Root Mean Squared Error : {rmse_base:.4}'))

def better_than_baseline(actual, predicted):
    '''better_than_baseline(actual, predicted)
    returns eval_df with SSE, MSE, and RMSE as rows and model_error, base_error, and model_efficacy as columns
    model_efficacy = True when model_error < base_error'''
    sse_res = sum((actual - predicted)  ** 2)
    ess_res = sum((predicted - actual.mean()) ** 2)
    tss_res = sse_res + ess_res
    mse_res = sse_res / len(actual)
    rmse_res = mse_res ** (1/2)
    baseline = actual.mean()
    sse_base = sum((actual - baseline)  ** 2)
    mse_base = sse_base / len(actual)
    rmse_base = mse_base ** (1/2)
    eval_df = pd.DataFrame(np.array(['SSE','MSE','RMSE']), columns=['metric'])
    eval_df['model_error'] = np.array([sse_res, mse_res, rmse_res])
    eval_df['base_error'] = np.array([sse_base, mse_base, rmse_base])
    eval_df['model_efficacy'] = eval_df['model_error'] < eval_df['base_error']
    return eval_df

def regression_stats(x, y):
    '''regression_stats(x, y)
    Returns slope, intercept, r, p, standard_error using stats.lineregress()
    '''
    slope, intercept, r, p, se = stats.linregress(x, y)
    return(print(f'|Slope of Regression: {slope:.4}'),
            print('|-----------------------------------'),
            print(f'|Regression intercept: {intercept:.4}'),
            print('|-----------------------------------'),
            print(f'|r\N{SUPERSCRIPT TWO} = {r ** 2:.2}'),
            print('|-----------------------------------'),
            print(f'|p value = {p:.4}'),
            print('|-----------------------------------'),
            print(f'|Standard error : {se:.4}'))