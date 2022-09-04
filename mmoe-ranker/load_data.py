import os
import pickle
import random
import numpy as np
import pandas as pd
from typing import Tuple, List

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


def create_negative_transactions(transactions_df: pd.DataFrame,
                                 article_df: pd.DataFrame) -> pd.DataFrame:
    all_article_ids = set(article_df['article_id'].unique())
    by_customer_df = transactions_df.groupby("customer_id", as_index=False)
    articles_by_customer = by_customer_df['article_id'].agg(['unique']).reset_index()
    articles_by_customer.columns = ['customer_id', 'article_list']

    negatives = []
    for _, row in articles_by_customer.iterrows():
        customer_id = row['customer_id']
        unpurchased_articles = list(all_article_ids - set(row['article_list']))
        sampled_article_ids = random.sample(unpurchased_articles, len(row['article_list']))
        for article_id in sampled_article_ids:
            negatives.append((customer_id, article_id, 0.0))
    return pd.DataFrame(negatives, columns=['customer_id', 'article_id', Features.LABEL])


def enrich_transactions(article_df: pd.DataFrame,
                        customer_df: pd.DataFrame,
                        transactions_df: pd.DataFrame):
    minimal_cust_df = customer_df[Features.CUSTOMER_FEATURES]
    minimal_art_df = article_df[Features.ARTICLE_FEATURES]
    cust_data = minimal_cust_df.set_index('customer_id').to_dict('index')
    art_data = minimal_art_df.set_index('article_id').to_dict('index')
    for cat_var in Features.CUSTOMER_FEATURES:
        if cat_var != 'customer_id':
            transactions_df[cat_var] = transactions_df['customer_id'].apply(lambda _id: cust_data[_id][cat_var])
    for cat_var in Features.ARTICLE_FEATURES:
        if cat_var != 'article_id':
            transactions_df[cat_var] = transactions_df['article_id'].apply(lambda _id: art_data[_id][cat_var])


def engineer_customer_features(transac_df: pd.DataFrame):
    # Engineer features by customer
    customer_transactions = transac_df.groupby("customer_id", as_index=False)
    customer_features = customer_transactions.agg('size')
    customer_features = customer_features.rename(columns={'size': 'cust_nb_transactions'})
    customer_features['cust_nb_dates'] = customer_transactions['t_dat'].nunique()['t_dat']
    for key in Features.ARTICLE_CATEG_FEATURES:
        # Add counter feature
        nb_categ_transactions = customer_transactions[key].nunique()[key]
        customer_features['cust_nb_' + key] = nb_categ_transactions
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
    article_features['art_nb_dates'] = article_transactions['t_dat'].nunique()['t_dat']
    for key in Features.CUSTOMER_CATEG_FEATURES:
        # Add counter feature
        nb_categ_transactions = article_transactions[key].nunique()[key]
        article_features['art_nb_' + key] = nb_categ_transactions
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


def merge_cross_features(customer_features: pd.DataFrame,
                         article_features: pd.DataFrame,
                         transac_df: pd.DataFrame):
    result_df = transac_df.merge(customer_features, on='customer_id', how='left')
    result_df = result_df.merge(article_features, on='article_id', how='left')
    return result_df


def augment_with_negative_examples(pos_transactions_df):
    neg_transactions_df = pos_transactions_df.copy()
    # We simply shuffle the customer id column to create the negative observations
    neg_transactions_df['customer_id'] = neg_transactions_df['customer_id'].sample(frac=1.0).to_list()
    neg_transactions_df[Features.LABEL] = 0.0
    all_transactions_df = pd.concat([pos_transactions_df, neg_transactions_df])
    # Shuffle positive and negative examples
    return all_transactions_df.sample(frac=1.0)


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
        negatives = random.sample(available_negs, 300)
        for neg_article_id in negatives:
            positive_obs = (customer_id, neg_article_id, 0.0)
            observations.append(positive_obs)
    return pd.DataFrame.from_records(observations, columns=['customer_id', 'article_id', Features.LABEL])


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if os.path.exists('train_df.p') and os.path.exists('test_df.p'):
        print('Loading existing train and test dataframes')
        train_df = pickle.load(open('train_df.p', 'rb'))
        test_df = pickle.load(open('test_df.p', 'rb'))
        customer_features = ['cust_nb_transactions', 'cust_nb_dates',
                             'cust_nb_article_id', 'cust_ratio_article_id',
                             'cust_nb_product_type_name', 'cust_ratio_product_type_name',
                             'cust_nb_product_group_name', 'cust_ratio_product_group_name',
                             'cust_nb_colour_group_name', 'cust_ratio_colour_group_name',
                             'cust_nb_department_name', 'cust_ratio_department_name',
                             'cust_nb_index_name', 'cust_ratio_index_name', 'cust_nb_section_name',
                             'cust_ratio_section_name', 'cust_nb_garment_group_name',
                             'cust_ratio_garment_group_name',
                             'cust_price_min', 'cust_price_max', 'cust_price_mean', 'cust_price_std']
        article_features = ['art_nb_transactions',
                            'art_nb_dates', 'art_nb_customer_id', 'art_ratio_customer_id',
                            'art_nb_FN', 'art_ratio_FN', 'art_nb_Active', 'art_ratio_Active',
                            'art_nb_club_member_status', 'art_ratio_club_member_status',
                            'art_nb_fashion_news_frequency', 'art_ratio_fashion_news_frequency',
                            'art_nb_age_interval', 'art_ratio_age_interval', 'art_nb_postal_code',
                            'art_ratio_postal_code', 'art_age_min', 'art_age_max', 'art_age_mean',
                            'art_age_std', 'art_price_min', 'art_price_max', 'art_price_mean', 'art_price_std']
        engineered_features = customer_features + article_features
        return train_df, test_df, engineered_features

    print('train and test dataframes not found on disk. Loading data...')

    print('Loading articles')
    article_df = pd.read_csv("../hmdata/articles.csv.zip")
    for categ_variable in Features.ARTICLE_CATEG_FEATURES:
        article_df[categ_variable] = article_df[categ_variable].astype(str)

    print('Loading customers')
    customer_df = pd.read_csv("../hmdata/customers.csv.zip")
    preprocess_customer_data(customer_df)
    for categ_variable in Features.CUSTOMER_CATEG_FEATURES:
        customer_df[categ_variable] = customer_df[categ_variable].astype(str)

    print('Loading transactions')
    transactions_df = pd.read_csv('../hmdata/transactions_train.csv.zip')
    transactions_df['article_id'] = transactions_df['article_id'].astype(str)
    transactions_df['customer_id'] = transactions_df['customer_id'].astype(str)

    date_feat = transactions_df['t_dat']
    print('Keep last 3 months of transactions only')
    past_three_month_transactions_df = transactions_df[date_feat >= '2020-06-09']
    print('Enrich transactions')
    enrich_transactions(article_df, customer_df, past_three_month_transactions_df)

    last_week_transactions_df = past_three_month_transactions_df[date_feat >= '2020-09-16']
    previous_week_transactions_df = past_three_month_transactions_df[(date_feat >= '2020-09-09') & (date_feat < '2020-09-16')]
    transactions_before_last_weeks_df = past_three_month_transactions_df[date_feat < '2020-09-16']
    all_candidate_articles = set(transactions_before_last_weeks_df.article_id.unique())
    train_df = build_dataset(all_candidate_articles, previous_week_transactions_df).sample(n=8_000_000)
    enrich_transactions(article_df, customer_df, train_df)
    test_df = build_dataset(all_candidate_articles, last_week_transactions_df).sample(n=200_000)
    enrich_transactions(article_df, customer_df, test_df)

    print('Engineer new features')
    article_features = engineer_article_features(transactions_before_last_weeks_df)
    customer_features = engineer_customer_features(transactions_before_last_weeks_df)
    train_df = merge_cross_features(customer_features, article_features, train_df)
    test_df = merge_cross_features(customer_features, article_features, test_df)

    engineered_features = list(customer_features.columns) + list(article_features.columns)

    for feature in engineered_features:
        test_df[feature].fillna(0.0, inplace=True)

    print('Save training and testing data on disk')
    pickle.dump(train_df, open('train_df.p', 'wb'))
    pickle.dump(test_df, open('test_df.p', 'wb'))

    return train_df, test_df, engineered_features
