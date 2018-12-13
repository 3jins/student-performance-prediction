import os
import torch
import torch.utils.data
from config import Config
from dataset import StudentData


class StudentDataLoader(torch.utils.data.DataLoader):
    config = Config.instance()
    data_root = os.path.join('..', 'data')

    def __init__(self, batch_size, subject, training_mode):
        self.batch_size = batch_size
        self.data = StudentData(
            data_root=self.data_root,
            subject=subject,
            training_mode=training_mode,
        )
        self.cuda = torch.cuda.is_available()
        additional_options = {'num_workers': 4, 'pin_memory': True} if self.cuda else {}
        super(StudentDataLoader, self).__init__(
            dataset=self.data,
            batch_size=batch_size,
            shuffle=self.config.SHUFFLE,
            **additional_options,
        )
