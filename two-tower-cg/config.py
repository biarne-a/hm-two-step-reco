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
