#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns


# numerical columns to drop
NUM_DROP = ['id', 'listing_url', 'scrape_id', 'host_id', 'host_listings_count',
            'host_total_listings_count', 'latitude', 'longitude', 
            'availability_30', 'availability_60', 'availability_90',
            'availability_365', 'calculated_host_listings_count',
            'security_deposit', 'reviews_per_month']
# categorical values to drop
CAT_DROP = ['host_since', 'host_location', 'street', 'neighbourhood_cleansed',
            'neighbourhood_group_cleansed', 'zipcode', 'country_code',
            'requires_license', 'host_verifications', 'market',
            'smart_location', 'amenities', 'calendar_updated', 'last_review',
            'has_availability', 'country', 'last_scraped', 'name', 'summary',
            'space', 'description', 'experiences_offered',
            'neighborhood_overview', 'notes', 'transit', 'city', 'state',
            'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url',
            'host_neighbourhood', 'host_url', 'host_name','host_about',
            'host_thumbnail_url', 'host_picture_url','calendar_last_scraped',
            'first_review', 'jurisdiction_names']

def coef_weights(coefficients, X_train):
    '''
    INPUT:
    coefficients - the coefficients of the linear model 
    X_train - the training data, so the column names can be used
    OUTPUT:
    coefs_df - a dataframe holding the coefficient, estimate, and abs(estimate)
    '''
    coefs_df = pd.DataFrame()
    coefs_df['est_int'] = X_train.columns
    coefs_df['coefs'] = coefficients
    coefs_df['abs_coefs'] = np.abs(coefficients)
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)
    return coefs_df


def create_dummy_df(df, cat_cols, dummy_na):
    '''
    INPUT:
    df - pandas dataframe with categorical variables you want to dummy
    cat_cols - list of strings that are associated with names of the categorical columns
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not
    
    OUTPUT:
    df - a new dataframe that has the following characteristics:
            1. contains all columns that were not specified as categorical
            2. removes all the original columns in cat_cols
            3. dummy columns for each of the categorical columns in cat_cols
            4. if dummy_na is True - it also contains dummy columns for the NaN values
            5. Use a prefix of the column name with an underscore (_) for separating 
    '''
    for col in  cat_cols:
        try:
            # for each cat add dummy var, drop original column
            df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=dummy_na)], axis=1)
        except:
            continue
    return df

def remove_much_nans(df, rate_max=0.7):
    cols_to_drop = set(df.columns[df.isnull().mean()>rate_max])
    print("dropping columns because of to many NaN Values:", cols_to_drop)
    df = df.drop(columns=cols_to_drop)
    return df

def clean_data(df, response_value, extra_drop_for_X):
    '''
    INPUT
    df - pandas dataframe 
    
    OUTPUT
    X - A matrix holding all of the variables you want to consider when predicting the response
    y - the corresponding response vector

    '''
    # EXTRA: make the prices a float64 type
    df['price'] = df['price'].str.replace(',','').str.replace('$','').astype('float')
    df['weekly_price'] = df['weekly_price'].str.replace(',','').str.replace('$','').astype('float')
    df['monthly_price'] = df['monthly_price'].str.replace(',','').str.replace('$','').astype('float')
    df['extra_people'] = df['extra_people'].str.replace(',','').str.replace('$','').astype('float')
    df['cleaning_fee'] = df['cleaning_fee'].str.replace(',','').str.replace('$','').astype('float')
    df['security_deposit'] = df['security_deposit'].str.replace(',','').str.replace('$','').astype('float')
    # make also float
    df['host_response_rate'] = df['host_response_rate'].str.replace('%','').astype('float')
    df['host_acceptance_rate'] = df['host_acceptance_rate'].str.replace('%','').astype('float')

    # 1-4:
    # remove rows where response value is missing
    df = df.dropna(subset=[response_value], axis=0)
    # remove columns without any values
    df = df.dropna(how='all', axis=1)
    # drop not useful columns for prediction
    df = df.drop(columns=NUM_DROP, axis=1)
    df = df.drop(columns=CAT_DROP, axis=1)
    df = remove_much_nans(df)
    
    # drop data which confuses the prediction / has not much meaning because of missing data
    num_df = df.select_dtypes(exclude=['object'])
    num_columns = num_df.columns
    df = df.dropna(subset=num_columns, thresh=len(num_columns)-2)
    df.to_csv("after_thresh_drop_dataframe.csv")

    # split off response dataframe
    y = df[response_value]

    # plot numerical graphs
    pre_plot(num_df)

    # drop response values from future X
    extra_drop_for_X.append(response_value) # drop also values which might be related to response value
    print(f"Excluding {extra_drop_for_X} from the model.")
    df = df.drop(columns=extra_drop_for_X, axis=1)

    # fill remaining NaN values of numerical columns with mean
    num_df = df.select_dtypes(exclude=['object'])
    num_columns = num_df.columns
    for col in num_columns:
        df[col].fillna((df[col].mean()), inplace=True)

    # take care of categorical columns
    cat_columns = df.select_dtypes(include=['object']).columns
    df = create_dummy_df(df, cat_columns, dummy_na=True)

    # save for overviiew of final dataframe
    df.to_csv("X_dataframe.csv")
    
    X = df

    return X, y

def pre_plot(df):
    df.hist()
    plt.show()
    sns.heatmap(df.corr(), annot=True, fmt=".2f")
    plt.show()

def fit_train_test_model(X, y, test_size=0.3, rand_state=42):
    # only use columns where more than a certain number of values are provided
    cutoff = 40
    X = X.iloc[:, np.where((X.sum() > cutoff) == True)[0]]

    #Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state)

    lm_model = LinearRegression(normalize=True) # Instantiate
    lm_model.fit(X_train, y_train) #Fit

    #Predict using your model
    y_test_preds = lm_model.predict(X_test)
    y_train_preds = lm_model.predict(X_train)

    #Score using your model
    test_score = mean_squared_error(y_test, y_test_preds)
    train_score = mean_squared_error(y_train, y_train_preds)

    return test_score, train_score, lm_model, X_train, X_test, y_train, y_test

def main():
    """
    This analysis aims to answer the following questions (Business Understanding):
    1. What are the key factors determining the price of the appartments?
    2. Will I be happier if I spend those extra dollars for the upgrade?
    
    For that, the price and rating scores of Airbnbs are used as a response
    value of a linear regression model. After training and testing the models, 
    taking a look at the coefficients of the model should lead to answering to
    above questions.
    This can bring us a better Business Understanding through Data Science
    while following the CRISP-DM process (steps are commented in code)!
    """
    df = pd.read_csv("data/listings.csv")
    # get an insight of the data --> CRISP-DM Data Understanding
    pre_plot(df)

    print("##################### Price")
    # --> CRISP-DM Data Preparation
    X,y = clean_data(df, "price", ['weekly_price', 'monthly_price'])
    #--> CRISP-DM Modeling and Evaluation 
    test_score, train_score, lm_model, X_train, X_test, y_train, y_test = fit_train_test_model(X, y)
    print("The score on the test data is: ", test_score)
    print("The score on the training data is: ", train_score)

    print("These are the highest and lowest 20 coefficients:")
    coef_df = coef_weights(lm_model.coef_, X_train)
    print(coef_df.head(20))
    print(coef_df.tail(20))
    
    print("##################### Price w/o neighbourhood")
    df = pd.read_csv("data/listings.csv")
    # --> CRISP-DM Data Preparation
    X,y = clean_data(df, "price", ['weekly_price', 'monthly_price', 'neighbourhood'])
    #--> CRISP-DM Modeling and Evaluation 
    test_score, train_score, lm_model, X_train, X_test, y_train, y_test = fit_train_test_model(X, y)
    print("The score on the test data is: ", test_score)
    print("The score on the training data is: ", train_score)

    print("These are the highest and lowest 20 coefficients:")
    coef_df = coef_weights(lm_model.coef_, X_train)
    print(coef_df.head(20))
    print(coef_df.tail(20))
    
    print("##################### Ratings")
    df = pd.read_csv("data/listings.csv")
    # --> CRISP-DM Data Preparation
    X,y = clean_data(df, 'review_scores_rating', ['review_scores_accuracy', 
        'review_scores_cleanliness', 'review_scores_checkin',
        'review_scores_communication', 'review_scores_location',
        'review_scores_value'])
    #--> CRISP-DM Modeling and Evaluation 
    test_score, train_score, lm_model, X_train, X_test, y_train, y_test = fit_train_test_model(X, y)
    print("The score on the test data is: ", test_score)
    print("The score on the training data is: ", train_score)

    print("These are the highest and lowest 20 coefficients:")
    coef_df = coef_weights(lm_model.coef_, X_train)
    print(coef_df.head(20))
    print(coef_df.tail(20))

if __name__ == "__main__":
    main()
