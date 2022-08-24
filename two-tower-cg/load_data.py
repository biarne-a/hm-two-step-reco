import os
import pickle
import pandas as pd
from typing import Tuple

from config import Variables


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


def split_data(transactions_df):
    trans_date = transactions_df['t_dat']
    train_df = transactions_df[(trans_date >= '2019-09-20') & (trans_date <= '2020-08-20')]
    test_df = transactions_df[trans_date >= '2020-08-20']
    return train_df, test_df


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    article_df = pd.read_csv("hmdata/articles.csv.zip")
    for categ_variable in Variables.ARTICLE_CATEG_VARIABLES:
        article_df[categ_variable] = article_df[categ_variable].astype(str)

    # if os.path.exists('train_df.p') and os.path.exists('test_df.p'):
    #     train_df = pickle.load(open('train_df.p', 'rb'))
    #     test_df = pickle.load(open('test_df.p', 'rb'))
    #     return train_df, test_df, article_df

    # article_df = pd.read_csv("hmdata/articles.csv.zip")
    customer_df = pd.read_csv("hmdata/customers.csv.zip")
    transactions_df = pd.read_csv('hmdata/transactions_train.csv.zip')
    # article_df['article_id'] = article_df['article_id'].astype(str)
    transactions_df['article_id'] = transactions_df['article_id'].astype(str)

    preprocess_customer_data(customer_df)
    minimal_trans_df = transactions_df[['article_id', 'customer_id', 't_dat']]
    minimal_cust_df = customer_df[Variables.CUSTOMER_CATEG_VARIABLES]
    minimal_art_df = article_df[Variables.ARTICLE_CATEG_VARIABLES]
    transactions_enhanced_df = minimal_trans_df.merge(minimal_cust_df, on='customer_id')
    transactions_enhanced_df = transactions_enhanced_df.merge(minimal_art_df, on='article_id')

    for categ_variable in Variables.ALL_CATEG_VARIABLES:
        transactions_enhanced_df[categ_variable] = transactions_enhanced_df[categ_variable].astype(str)

    train_df, test_df = split_data(transactions_enhanced_df)
    pickle.dump(train_df, open('train_df.p', 'wb'))
    pickle.dump(test_df, open('test_df.p', 'wb'))

    return train_df, test_df, article_df
