from typing import List


class Config:
    def __init__(self,
                 embedding_dimension: int,
                 batch_size: int,
                 learning_rate: float,
                 nb_epochs: int):
        self.embedding_dimension = embedding_dimension
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.nb_epochs = nb_epochs

    def to_json(self):
        return {
            'embedding_dimension': self.embedding_dimension,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'nb_epochs': self.nb_epochs
        }


class Variables:
    ARTICLE_CATEG_VARIABLES: List[str] = ['article_id', 'product_type_name', 'product_group_name', 'colour_group_name',
                                          'department_name', 'index_name', 'section_name', 'garment_group_name']
    CUSTOMER_CATEG_VARIABLES: List[str] = ['customer_id', 'FN', 'Active', 'club_member_status',
                                           'fashion_news_frequency', 'age_interval', 'postal_code']
    ALL_CATEG_VARIABLES: List[str] = ARTICLE_CATEG_VARIABLES + CUSTOMER_CATEG_VARIABLES
