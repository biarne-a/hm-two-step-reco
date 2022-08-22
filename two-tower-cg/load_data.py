import pandas as pd


class HmData:
    def __init__(self,
                 article_df: pd.DataFrame,
                 customer_df: pd.DataFrame,
                 transactions_df: pd.DataFrame):
        self.article_df = article_df
        self.customer_df = customer_df
        self.transactions_df = transactions_df


def load_data() -> HmData:
    article_df = pd.read_csv("hmdata/articles.csv.zip")
    customer_df = pd.read_csv("hmdata/customers.csv.zip")
    transactions_df = pd.read_csv('hmdata/transactions_train.csv.zip')
    article_df['article_id'] = article_df['article_id'].astype(str)
    transactions_df['article_id'] = transactions_df['article_id'].astype(str)

    return HmData(article_df, customer_df, transactions_df)