import os
from config import Config
from data_loader import StudentDataLoader
from dataset import StudentData
from model import NNModel
from data_explorer import Trainer, Evaluator

if __name__ == '__main__':
    config = Config.instance()
    trainer = Trainer()
    trainer.train(config.MAX_EPOCH)
    evaluator = Evaluator()
    evaluator.evaluate()
