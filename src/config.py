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
        self.MAX_EPOCH = 100
        self.BATCH_SIZE = 32
        # self.CRITERION = torch.nn.CrossEntropyLoss()

        # Model
        self.INPUT_SIZE = 30
        self.HIDDEN_SIZE = [512]
        self.OUTPUT_SIZE = 3

        # Data
        self.MAT_TEST_SET_SIZE = 50
        self.POR_TEST_SET_SIZE = 100
        self.SHUFFLE = True
