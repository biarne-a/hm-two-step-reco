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


def split_data(df):
    trans_date = df['t_dat']
    train_df = df[(trans_date >= '2019-09-20') & (trans_date <= '2020-08-20')]
    test_df = df[trans_date >= '2020-08-20']
    return train_df, test_df


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if os.path.exists('train_df.p') and os.path.exists('test_df.p') and os.path.exists('article_df.p'):
        train_df = pickle.load(open('train_df.p', 'rb'))
        test_df = pickle.load(open('test_df.p', 'rb'))
        article_df = pickle.load(open('article_df.p', 'rb'))
        return train_df, test_df, article_df

    article_df = pd.read_csv("hmdata/articles.csv.zip")
    for categ_variable in Variables.ARTICLE_CATEG_VARIABLES:
        article_df[categ_variable] = article_df[categ_variable].astype(str)

    customer_df = pd.read_csv("hmdata/customers.csv.zip")
    preprocess_customer_data(customer_df)
    for categ_variable in Variables.CUSTOMER_CATEG_VARIABLES:
        customer_df[categ_variable] = customer_df[categ_variable].astype(str)

    transactions_df = pd.read_csv('hmdata/transactions_train.csv.zip')
    transactions_df['article_id'] = transactions_df['article_id'].astype(str)
    transactions_df['customer_id'] = transactions_df['customer_id'].astype(str)

    transactions_df = transactions_df[['article_id', 'customer_id', 't_dat']]
    minimal_cust_df = customer_df[Variables.CUSTOMER_CATEG_VARIABLES]
    minimal_art_df = article_df[Variables.ARTICLE_CATEG_VARIABLES]
    cust_data = minimal_cust_df.set_index('customer_id').to_dict('index')
    art_data = minimal_art_df.set_index('article_id').to_dict('index')
    for cat_var in Variables.CUSTOMER_CATEG_VARIABLES:
        if cat_var != 'customer_id':
            transactions_df[cat_var] = transactions_df['customer_id'].apply(lambda _id: cust_data[_id][cat_var])
    for cat_var in Variables.ARTICLE_CATEG_VARIABLES:
        if cat_var != 'article_id':
            transactions_df[cat_var] = transactions_df['article_id'].apply(lambda _id: art_data[_id][cat_var])

    train_df, test_df = split_data(transactions_df)
    pickle.dump(train_df, open('train_df.p', 'wb'))
    pickle.dump(test_df, open('test_df.p', 'wb'))
    pickle.dump(article_df, open('article_df.p', 'wb'))

    return train_df, test_df, article_df
