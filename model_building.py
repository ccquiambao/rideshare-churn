import pandas as pd
import numpy as np
import datetime as dt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def df_target_creation(df):
    '''
    INPUT: df, original dataframe
    OUTPUT: df, w/ target added and features used in target creation dropped
    '''
    dt_cols = ['last_trip_date']
    for col in dt_cols:
        df[col] = pd.to_datetime(df[col])

    churn_date = df['last_trip_date'].max() + dt.timedelta(-30)
    df['Churned'] = (df['last_trip_date'] <= churn_date)
    df.drop(['last_trip_date', 'signup_date'], axis=1, inplace=True)

    return df

def df_dummify_cols(df):
    '''
    INPUT: df, w/ undummified cols
           list, of categorical cols
           list, refernce categories to drop
    OUTPUT: df, w/ dummified cols and reference category dropped
    '''
    cat_cols = ['city', 'phone']
    ref_cols = ['city_Winterfell', 'phone_Android']

    df['phone'].fillna('other', inplace=True) # fill in missing values w/ new cat
    df = pd.get_dummies(df, columns=cat_cols)
    df.drop(ref_cols, axis=1, inplace=True)

    return df


def df_create_features(df):
    '''
    INPUT: df, w/ number of trips in first 30 days
    OUTPUT: df, w/ added columns for exactly 1, 0, or 1 or 0 trips, rating is five
    '''
    df['one_trip_first_thirty'] = df['trips_in_first_30_days'] == 1
    df['zero_trip_first_thirty'] = df['trips_in_first_30_days'] == 0
    df['zero_one_trip_first_thirty'] = (df['one_trip_first_thirty'] |
                                       df['zero_trip_first_thirty'])
    df['five_by_driver'] = df['avg_rating_by_driver'] == 5
    df['five_of_driver'] = df['avg_rating_of_driver'] == 5

    return df

def df_preprocessing(df):
    '''
    There's a better way to do this
    '''
    df = df_target_creation(df)
    df = df_dummify_cols(df)
    df = df_create_features(df)

    return df

class RatingsImputer(BaseEstimator, TransformerMixin):
    '''
    INPUT: df, w/ ratings cols having missing vals
    OUTPUT: df, w/ ratings cols imputed w/ mean rating and imputed col
    '''

    rating_dict = {'avg_rating_by_driver': 0.,
                   'avg_rating_of_driver': 0.}

    def fit(self, X, y):
        for col in self.rating_dict.iterkeys():
            self.rating_dict[col] = X[col].mean()

        return self

    def transform(self, X):
        X['missing_by_driver'] = X['avg_rating_by_driver'].isnull()
        X['missing_of_driver'] = X['avg_rating_of_driver'].isnull()
        print X.columns
        for col, col_mean in self.rating_dict.iteritems():
            X[col].fillna(col_mean, inplace=True)


        return X


# TODO create feature_selection class for model
# TODO complete crossvalidation pipeline
# TODO create models
if __name__ == '__main__':
    rideshare_df = pd.read_csv('data/churn.csv')
    rideshare_df = df_preprocessing(rideshare_df)
    y = rideshare_df.pop('Churned')
    X = rideshare_df

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    X_train.columns
    pipe = Pipeline([
            ('ratings', RatingsImputer()),
            ('scaler', StandardScaler()),
            ('logit', LogisticRegression())
                    ])

    scores = []
    kf = KFold(n_splits=5, random_state=42)
    for train_index, val_index in kf.split(X_train):
        cv_X = X_train.copy()
        cv_y = y_train.copy()
        X_t, X_v = cv_X.iloc[train_index, :], cv_X.iloc[val_index, :]
        y_t, y_v = cv_y[train_index], cv_y[val_index]
        pipe.fit(X_t, y_t)
        score = pipe.score(X_v, y_v)
        scores.append(score)
