import pandas as pd
import numpy as np
import datetime as dt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


def df_preprocessing(df, dt_cols):
    '''
    INPUT: df, list of columns to be converted to datetime
    OUTPUT: df, with columns converted to datetime
    '''

    for col in dt_cols:
        df[col] = pd.to_datetime(df[col])

    return df


def df_target_creation(df):
    '''
    INPUT: df, original dataframe
    OUTPUT: df, w/ target added and features used in target creation dropped
    '''

    churn_date = df['last_trip_date'].max() + dt.timedelta(-30)
    df['Churned'] = df['last_trip_date'] <= churn_date
    df.drop(['last_trip_date'], axis=1, inplace=True)

    return df

def df_dummify(df, cat_cols, ref_cols):
    '''
    INPUT: df, w/ undummified cols
           list, of categorical cols
           list, refernce categories to drop
    OUTPUT: df, w/ dummified cols and reference category dropped
    '''
    df['phone'].fillna('other', inplace=True) # fill in missing values w/ new cat
    df = pd.get_dummies(df, columns=cat_cols)
    df.drop(ref_cols, axis=1, inplace=True)

    return df

def df_weekday_pct_features():
    rideshare_df['one_trip_first_thirty'] = rideshare_df['trips_in_first_30_days'] == 1
    rideshare_df['zero_trip_first_thirty'] = rideshare_df['trips_in_first_30_days'] == 0
    rideshare_df['zero_one_trip_first_thirty'] = (rideshare_df['one_trip_first_thirty'] |
                                                  rideshare_df['zero_trip_first_thirty'])



# TODO create imputing class for ratings
# TODO create feature_selection class for model
# TODO complete crossvalidation pipeline
# TODO create models
if __name__ == '__main__':
    rideshare_df = pd.read_csv('data/churn.csv')

    dt_cols = ['last_trip_date', 'signup_date']
    rideshare_df = df_preprocessing(rideshare_df, dt_cols)
    rideshare_df = df_target_creation(rideshare_df)

    cat_cols = ['city', 'phone']
    ref_cols = ['city_Winterfell', 'phone_Android']
    rideshare_df = df_dummify(rideshare_df, cat_cols, ref_cols)

    df_weekday_pct_features()

    # data to train simple model
    y = rideshare_df.pop('Churned').values.reshape(-1, 1)
    X = rideshare_df['surge_pct'].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # train simple model
    logit = LogisticRegression()
    logit.fit(X_train, y_train)
