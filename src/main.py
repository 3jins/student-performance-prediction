from config import Config
from data_loader import StudentDataLoader
from dataset import StudentData
from model import NNModel
from trainer import Trainer

if __name__ == '__main__':
    config = Config.instance()

    training_data = StudentData(
        data_root='../data',
        subject='mat',
        is_training_mode=True,
    )
    training_model = NNModel(is_training_mode=True)
    training_data_loader = StudentDataLoader(batch_size=config.BATCH_SIZE, is_training_mode=True)
    trainer = Trainer(training_model, training_data_loader)
    trainer.train(config.MAX_EPOCH)

    test_data = StudentData(
        data_root='../data',
        subject='mat',
        is_training_mode=False,
    )
    test_model = NNModel(is_training_mode=False)
    test_data_loader = StudentDataLoader(batch_size=config.BATCH_SIZE, is_training_mode=False)
    evaluater = Trainer(test_model, test_data_loader)
    evaluater.evaluate()
