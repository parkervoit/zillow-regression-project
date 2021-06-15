# Predicting Property Tax Rates with Zillow
![](https://searchlogovector.com/wp-content/uploads/2018/10/zillow-logo-vector.png)
### Description
> The purpose of this project is to create a Linear Regression model to predict property tax values and present results to the Zillow Data Science Team. 
### Project Goals
- Create a machine learning model to predict property values of single unit properites using houses sold during may - august 2017
- Identify where the states and counties the properties are located in. (can be done with FIPS data)
- Present results to the Zillow data science team
#### Deliverables
1. Git hub repository with analysis and work
2. Jupyter Notebook detailing analytical process and decisions
3. [Slide deck](https://www.canva.com/design/DAEhcAH6u_E/2CU4r1H2SkcqZ_KEcWUXMA/view?utm_content=DAEhcAH6u_E&utm_campaign=designshare&utm_medium=link&utm_source=sharebutton) with presentation summarizing findings
### Data Context and Dictionary
- Database is made up of Zillow property data pulled from the Codeup server. It can also be found [here](https://www.kaggle.com/c/zillow-prize-1) at Kaggle
Data Dictionary:
- bed : number of bedrooms in the house 
- bath : number of bathrooms in the house. It is measured in increments of .5
- sqft : calculated square footage of the house. This was used over other square footage columns due to the completeness of the column. 
- tax_value : the data target we are trying to predict. It is the assessed tax value in USD. 
- fips/county : info was found from the USDA [website](https://www.nrcs.usda.gov/wps/portal/nrcs/detail/national/home/?cid=nrcs143_013697). Codes are as follows:
    - 6037 = Los Angeles(LA) County, California
    - 6059 = Orange County, California
    - 6111 = Ventura County, California
### Planning
- Acquire single unit property data with a transaction date between May and August 2017
- Prep data by dropping duplicates, nulls, and data points more than 3 standard deviations away
- Will need to scale data, too
- Look up FIPS codes and calculate tax rate to determine distributions
- Use visualizations and stats to determine drivers of property cost
- Develop and test hypotheses
- Create baselines and choose best based off of RMSE
- Determine type of regression based off of target distribution
- Choose best performing model based off of findings such as RMSE and r2
- Improve on MVP through feature engineering and automated feature selection
- Communicate results with a slide deck
### Initial Hypotheses 
1. Tax rates will be correlated with square footage and as an extension bedroom and bathroom count
2. Tax rates will significantly vary between the counties
### Instructions for Reproducibility 
To recreate this project, you will need an env.py module with login credentials to the Codeup database or you can acquire the data from the kaggle website. You will also need to know how and able to use the pandas, seaborn, scipy.stats, and sklearn libraries. Finally, you will need the acquire, prepare, explore, evaluate modules contained within this git repo. 
