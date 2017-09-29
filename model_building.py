import pandas as pd
import numpy as np
import datetime as dt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import matplotlib.pyplot as plt


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
    # df['zero_one_trip_first_thirty'] = (df['one_trip_first_thirty'] |
    #                                    df['zero_trip_first_thirty'])
    # feature doesn't help
    df['five_by_driver'] = df['avg_rating_by_driver'] == 5
    df['five_of_driver'] = df['avg_rating_of_driver'] == 5

    return df

def df_preprocessing(df):
    '''
    INPUT: df, unaltered
    OUTPUT: df, processed
    '''
    # TODO do this better
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
        for col, col_mean in self.rating_dict.iteritems():
            X[col].fillna(col_mean, inplace=True)


        return X

class FeatureSelector(BaseEstimator, TransformerMixin):
    '''
    Select features to train model on
    '''
    def __init__(self, features=None):
        if features:
            self.features = features
        else:
            self.features = None
    def fit(self, X, y):
        return self

    def transform(self, X):
        if self.features:
            self.columns = X.loc[:, self.features].columns
            return X.loc[:, self.features]
        else:
            self.columns = X
            return X

def lasso_feature_selection(pipe):
    '''
    INPUT: pipeline class
    OUTPUT: Plot of beta coefficients over increasing values of regularization param
    '''
    c_vals = np.logspace(3, -2, num=100)
    beta_dict = defaultdict(list)
    for c in c_vals:
        train_set = X_train.copy()
        print c
        pipe.named_steps['logit'].set_params(penalty='l1')
        pipe.named_steps['logit'].set_params(C=c)
        pipe.fit(train_set, y_train)
        coeff_list = pipe.named_steps['logit'].coef_[0]
        for col, coeff in zip(train_set.columns, coeff_list):
            beta_dict[col].append(coeff)

    plt.figure(figsize=(20,12))
    plt.ylim(-1, 1)
    for col, beta in beta_dict.iteritems():
        plt.plot(1/c_vals, beta, label=col)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    rideshare_df = pd.read_csv('data/churn.csv')
    rideshare_df = df_preprocessing(rideshare_df)
    y = rideshare_df.pop('Churned')
    X = rideshare_df

    # drop these after lasso feature seelction
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    pipe = Pipeline([
            ('ratings', RatingsImputer()),
            ('features', FeatureSelector()),
            ('scaler', StandardScaler()),
            ('logit', LogisticRegression())
                    ])

    lasso_feature_selection(pipe)

    # grid search params
    penalties = ['l1', 'l2']
    c_vals = [0.01, 0.1, 1, 10, 100]
    feature_set_all = X_train.columns.tolist()
    feature_set_mid = ['avg_dist', 'surge_pct', 'luxury_car_user', 'city_Astapor',
                       'avg_rating_of_driver', 'avg_rating_by_driver',
                       "city_King's Landing", 'five_by_driver', 'five_of_driver']
    feature_set_mid_miss = ['avg_dist', 'surge_pct', 'luxury_car_user', 'city_Astapor',
                       'avg_rating_of_driver', 'avg_rating_by_driver',
                       "city_King's Landing", 'five_by_driver', 'five_of_driver',
                       'missing_of_driver', 'missing_by_driver']

    feature_sets = [feature_set_all, feature_set_mid, feature_set_mid_miss]

    # grid search
    # sklearn has an implementation, but preference is to script own
    scores_dict = dict()
    kf = KFold(n_splits=5, random_state=42)
    for feature_set in feature_sets:
        for penalty in penalties:
            for c in c_vals:

                pipe.named_steps['logit'].set_params(penalty=penalty)
                pipe.named_steps['logit'].set_params(C=c)

                key = (penalty, c, tuple(feature_set))
                print key
                counter = 0
                scores = []

                for train_index, val_index in kf.split(X_train):
                    counter += 1
                    print counter
                    train_set = X_train.loc[:, feature_set].copy()
                    X_t, X_v = (train_set.iloc[train_index, :],
                                train_set.iloc[val_index, :])
                    y_t, y_v = y_train[train_index], y_train[val_index]

                    pipe.fit(X_t, y_t)
                    score = pipe.score(X_v, y_v)
                    scores.append(score)

                scores_dict[key] = np.mean(score)

    best_params = max(scores_dict.items())
    print best_params
    penalty, c_val, feature_set = best_params[0]

    # train model w/ best params
    pipe.named_steps['features'].features = list(feature_set)
    pipe.named_steps['logit'].set_params(penalty=penalty)
    pipe.named_steps['logit'].set_params(C=c_val)

    pipe.fit(X_train.loc[:, feature_set], y_train)
    final_score = pipe.score(X_test.loc[:, feature_set], y_test)

    print 'test score: {}'.format(final_score)
    final_coeffs = pipe.named_steps['logit'].coef_[0]
    final_cols = pipe.named_steps['features'].columns
    col_coef = sorted(zip(final_cols, final_coeffs), key=lambda x: abs(x[1]))
    for col, coef in col_coef:
        print col, coef
