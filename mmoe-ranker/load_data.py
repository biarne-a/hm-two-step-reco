import os
import pickle
import random
import numpy as np
import pandas as pd
from typing import List

from features import Features


def create_age_interval(x):
    if x <= 25:
        return '[16, 25]'
    if x <= 35:
        return '[26, 35]'
    if x <= 45:
        return '[36, 45]'
    if x <= 55:
        return '[46, 55]'
    if x <= 65:
        return '[56, 65]'
    return '[66, 99]'


def preprocess_customer_data(customer_df):
    customer_df["FN"].fillna("UNKNOWN", inplace=True)
    customer_df["Active"].fillna("UNKNOWN", inplace=True)

    # Set unknown the club member status & news frequency
    customer_df["club_member_status"].fillna("UNKNOWN", inplace=True)

    customer_df["fashion_news_frequency"] = customer_df["fashion_news_frequency"].replace({"None": "NONE"})
    customer_df["fashion_news_frequency"].fillna("UNKNOWN", inplace=True)

    # Set missing values in age with the median
    customer_df["age"].fillna(customer_df["age"].median(), inplace=True)
    customer_df["age_interval"] = customer_df["age"].apply(lambda x: create_age_interval(x))


def split_data(df):
    trans_date = df['t_dat']
    train_df = df[(trans_date >= '2019-09-20') & (trans_date <= '2020-08-20')]
    test_df = df[trans_date >= '2020-09-16']
    return train_df, test_df


def enrich_transactions(article_df: pd.DataFrame,
                        customer_df: pd.DataFrame,
                        transactions_df: pd.DataFrame):
    result_df = transactions_df.merge(customer_df, on='customer_id')
    return result_df.merge(article_df, on='article_id')


def engineer_customer_features(transac_df: pd.DataFrame):
    # Engineer features by customer
    customer_transactions = transac_df.groupby("customer_id", as_index=False)
    customer_features = customer_transactions.agg('size')
    customer_features = customer_features.rename(columns={'size': 'cust_nb_transactions'})
    customer_features['cust_nb_dates'] = customer_transactions['t_dat'].nunique()['t_dat'].astype(np.float64)
    for key in Features.ARTICLE_CATEG_FEATURES:
        # Add counter feature
        nb_categ_transactions = customer_transactions[key].nunique()[key]
        customer_features['cust_nb_' + key] = nb_categ_transactions.astype(np.float64)
        # Add ratio feature
        customer_features['cust_ratio_' + key] = nb_categ_transactions / customer_features['cust_nb_transactions']
    for key in Features.ARTICLE_CONTI_FEATURES + Features.TRANSACTION_CONTI_FEATURES:
        # Add common statistics features
        customer_features['cust_' + key + '_min'] = customer_transactions[key].min()[key]
        customer_features['cust_' + key + '_max'] = customer_transactions[key].max()[key]
        customer_features['cust_' + key + '_mean'] = customer_transactions[key].mean()[key]
        customer_features['cust_' + key + '_std'] = customer_transactions[key].std()[key]
        customer_features['cust_' + key + '_std'].fillna(0.0, inplace=True)

    return customer_features


def engineer_article_features(transac_df: pd.DataFrame):
    # Engineer features by article
    article_transactions = transac_df.groupby("article_id", as_index=False)
    article_features = article_transactions.agg('size')
    article_features = article_features.rename(columns={'size': 'art_nb_transactions'})
    article_features['art_nb_transactions'] = article_features['art_nb_transactions'].astype(np.float64)
    article_features['art_nb_dates'] = article_transactions['t_dat'].nunique()['t_dat'].astype(np.float64)
    for key in Features.CUSTOMER_CATEG_FEATURES:
        # Add counter feature
        nb_categ_transactions = article_transactions[key].nunique()[key]
        article_features['art_nb_' + key] = nb_categ_transactions.astype(np.float64)
        # Add ratio feature
        article_features['art_ratio_' + key] = nb_categ_transactions / article_features['art_nb_transactions']
    for key in Features.CUSTOMER_CONTI_FEATURES + Features.TRANSACTION_CONTI_FEATURES:
        # Add common statistics features
        article_features['art_' + key + '_min'] = article_transactions[key].min()[key]
        article_features['art_' + key + '_max'] = article_transactions[key].max()[key]
        article_features['art_' + key + '_mean'] = article_transactions[key].mean()[key]
        article_features['art_' + key + '_std'] = article_transactions[key].std()[key]
        article_features['art_' + key + '_std'].fillna(0.0, inplace=True)

    return article_features


def replace_missing_values(df, engineered_article_columns, engineered_customer_columns):
    # Replace missing values
    for key in Features.ARTICLE_CONTI_FEATURES + Features.TRANSACTION_CONTI_FEATURES:
        df['cust_' + key + '_min'].fillna(df['cust_' + key + '_min'].median(), inplace=True)
        df['cust_' + key + '_max'].fillna(df['cust_' + key + '_max'].median(), inplace=True)
        df['cust_' + key + '_mean'].fillna(df['cust_' + key + '_mean'].median(), inplace=True)
    for key in engineered_customer_columns:
        df[key].fillna(0.0, inplace=True)

    # Replace missing values
    for key in Features.CUSTOMER_CONTI_FEATURES + Features.TRANSACTION_CONTI_FEATURES:
        df['art_' + key + '_min'].fillna(df['art_' + key + '_min'].median(), inplace=True)
        df['art_' + key + '_max'].fillna(df['art_' + key + '_max'].median(), inplace=True)
        df['art_' + key + '_mean'].fillna(df['art_' + key + '_mean'].median(), inplace=True)
    for key in engineered_article_columns:
        df[key].fillna(0.0, inplace=True)


def merge_cross_features(customer_features: pd.DataFrame,
                         article_features: pd.DataFrame,
                         transac_df: pd.DataFrame):
    result_df = transac_df.merge(customer_features, on='customer_id', how='left')
    result_df = result_df.merge(article_features, on='article_id', how='left')
    return result_df


def build_dataset(all_articles, previous_week_transactions_df):
    observations = []
    for customer_id, customer_transactions_df in previous_week_transactions_df.groupby('customer_id'):
        # Add positive observations
        for i, row in customer_transactions_df.iterrows():
            positive_obs = (customer_id, row['article_id'], 1.0)
            observations.append(positive_obs)
        # Add random negative observations
        customer_articles = set(customer_transactions_df.article_id.unique())
        available_negs = list(all_articles - customer_articles)
        negatives = random.sample(available_negs, 10)
        for neg_article_id in negatives:
            positive_obs = (customer_id, neg_article_id, 0.0)
            observations.append(positive_obs)
    return pd.DataFrame.from_records(observations, columns=['customer_id', 'article_id', Features.LABEL1])


class HmData:
    def __init__(self, article_df, customer_df, train_df, test_df,
                 all_articles_counts,
                 engineered_article_features: pd.DataFrame,
                 engineered_customer_features: pd.DataFrame):
        self.article_df = article_df
        self.customer_df = customer_df
        self.train_df = train_df
        self.test_df = test_df
        self.all_articles_counts = all_articles_counts
        self.engineered_article_features = engineered_article_features
        self.engineered_customer_features = engineered_customer_features

    @property
    def engineered_columns(self) -> List[str]:
        return self.engineered_article_columns + self.engineered_customer_columns

    @property
    def engineered_article_columns(self) -> List[str]:
        return list(set(self.engineered_article_features.columns) - {'article_id'})

    @property
    def engineered_customer_columns(self):
        return list(set(self.engineered_customer_features.columns) - {'customer_id'})


def find_unique_values(train_df: pd.DataFrame, article_df: pd.DataFrame, customer_df: pd.DataFrame):
    unique_values = {}

    print('Find unique article categorical values')
    transactions_x_article_data = train_df.merge(article_df, on='article_id')
    for categ_variable in Features.ARTICLE_CATEG_FEATURES:
        unique_values[categ_variable] = transactions_x_article_data[categ_variable].unique()

    print('Find unique customer categorical values')
    transactions_x_customer_data = train_df.merge(customer_df, on='customer_id')
    for categ_variable in Features.CUSTOMER_CATEG_FEATURES:
        unique_values[categ_variable] = transactions_x_customer_data[categ_variable].unique()

    return unique_values


def load_data() -> HmData:
    if os.path.exists('data.p'):
        print('Loading existing data')
        return pickle.load(open('data.p', 'rb'))

    print('data not found on disk. Loading data...')

    print('Loading articles')
    article_df = pd.read_csv("../hmdata/articles.csv.zip")[Features.ARTICLE_FEATURES]
    for categ_variable in Features.ARTICLE_CATEG_FEATURES:
        article_df[categ_variable] = article_df[categ_variable].astype(str)

    print('Loading customers')
    customer_df = pd.read_csv("../hmdata/customers.csv.zip")
    preprocess_customer_data(customer_df)
    for categ_variable in Features.CUSTOMER_CATEG_FEATURES:
        customer_df[categ_variable] = customer_df[categ_variable].astype(str)
    customer_df = customer_df[Features.CUSTOMER_FEATURES]

    print('Loading transactions')
    transactions_df = pd.read_csv('../hmdata/transactions_train.csv.zip', parse_dates=['t_dat'])
    transactions_df['article_id'] = transactions_df['article_id'].astype(str)
    transactions_df['customer_id'] = transactions_df['customer_id'].astype(str)
    transactions_df['week'] = (transactions_df['t_dat'] - transactions_df['t_dat'].min()).dt.days // 7

    print('Use last 3 months of transactions only')
    last_week = transactions_df['week'].max()
    first_week = last_week - 7
    df = transactions_df[transactions_df['week'] >= first_week]
    rand_val = np.random.uniform(0, 1, size=len(df))

    print('Enrich transaction data with metadata')
    df = enrich_transactions(article_df, customer_df, df)
    # All transactions are positive examples. Negatives will be sampled later
    df[Features.LABEL1] = 1.0

    # Split: last week for test and the previous ones for training
    print('Split data into train and test data')
    # test_df = df.loc[df['week'] == last_week]
    # train_df = df.loc[(df['week'] >= first_week) & (df['week'] < last_week)]
    test_df = df[rand_val < 0.1]
    train_df = df[rand_val >= 0.1]

    customer_freq = train_df.customer_id.value_counts()
    warm_customer_ids = customer_freq[customer_freq > 5].index.tolist()
    train_df = train_df[train_df.customer_id.isin(warm_customer_ids)]
    test_df = test_df[test_df.customer_id.isin(warm_customer_ids)]

    # Build article counts map for popularity negative sampling
    train_freq = train_df.article_id.value_counts()
    all_articles_counts = train_freq.to_dict()
    warm_train_article_ids = train_freq[train_freq > 10].index.tolist()
    train_df = train_df[train_df.article_id.isin(warm_train_article_ids)]
    test_df = test_df[test_df.article_id.isin(warm_train_article_ids)]

    print('Engineer new features')
    article_features = engineer_article_features(train_df)
    customer_features = engineer_customer_features(train_df)

    print('Merge engineered features')
    train_df = merge_cross_features(customer_features, article_features, train_df)
    test_df = merge_cross_features(customer_features, article_features, test_df)

    data = HmData(article_df, customer_df, train_df, test_df, all_articles_counts,
                  article_features, customer_features)

    print('Replace missing test values')
    replace_missing_values(data.test_df, data.engineered_article_columns, data.engineered_customer_columns)

    print('Save data to disk')
    pickle.dump(data, open('data.p', 'wb'))

    return data
