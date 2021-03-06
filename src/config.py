import torch

class SingletonInstance:
    _instance = None

    @classmethod
    def __getInstance(cls):
        return cls._instance

    @classmethod
    def instance(cls, *args, **kargs):
        cls._instance = cls(*args, **kargs)
        cls.instance = cls.__getInstance
        return cls._instance


class Config(SingletonInstance):
    """Set Hyper-parameters of models in here.
    """

    def __init__(self):
        # Train
        self.LEARNING_RATE = 0.01
        self.MAX_EPOCH = 1000
        self.BATCH_SIZE = 32

        # Model
        self.INPUT_SIZE = 30
        self.HIDDEN_SIZES = [512, 128, 64]
        self.OUTPUT_SIZE = 3

        # Data
        self.SUBJECT = 'por'
        self.MAT_TEST_SET_SIZE = 50
        self.POR_TEST_SET_SIZE = 100
        self.SHUFFLE = True
