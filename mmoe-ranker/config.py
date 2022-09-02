class Config:
    def __init__(self,
                 batch_size: int,
                 learning_rate: float,
                 nb_epochs: int):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.nb_epochs = nb_epochs

    def to_json(self):
        return {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'nb_epochs': self.nb_epochs
        }
