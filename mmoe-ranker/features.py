from typing import List


class Features:
    ARTICLE_CATEG_FEATURES: List[str] = ['article_id', 'product_type_name', 'product_group_name', 'colour_group_name',
                                         'department_name', 'index_name', 'section_name', 'garment_group_name']
    ARTICLE_CONTI_FEATURES: List[str] = []
    ARTICLE_FEATURES = ARTICLE_CATEG_FEATURES + ARTICLE_CONTI_FEATURES

    CUSTOMER_CATEG_FEATURES: List[str] = ['customer_id', 'FN', 'Active', 'club_member_status',
                                          'fashion_news_frequency', 'age_interval', 'postal_code']
    CUSTOMER_CONTI_FEATURES: List[str] = ['age']
    CUSTOMER_FEATURES = CUSTOMER_CATEG_FEATURES + CUSTOMER_CONTI_FEATURES

    TRANSACTION_CONTI_FEATURES: List[str] = ['price']

    ALL_CATEG_FEATURES = ARTICLE_CATEG_FEATURES + CUSTOMER_CATEG_FEATURES
    ALL_CONTI_FEATURES = ARTICLE_CONTI_FEATURES + CUSTOMER_CONTI_FEATURES

    LABEL1 = 'bought'
    LABEL2 = 'product_type_name'
    LABEL3 = 'product_group_name'
    LABEL4 = 'department_name'
    LABEL5 = 'index_name'
    LABEL6 = 'section_name'
    LABEL7 = 'garment_group_name'

    ALL_VARIABLES: List[str] = ALL_CATEG_FEATURES + ALL_CONTI_FEATURES + [LABEL1]
