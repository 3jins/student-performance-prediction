import torch
import torch.utils.data
from config import Config
from dataset import StudentData


class StudentDataLoader(torch.utils.data.DataLoader):
    config = Config.instance()
    data_root = '../data'

    def __init__(self, batch_size, train=True, debug_mode=False, subject='mat', ):
        self.data = StudentData(
            data_root=self.data_root,
            train=train,
            debug_mode=debug_mode,
            subject=subject
        )
        self.cuda = torch.cuda.is_available()
        additional_options = {'num_workers': 4, 'pin_memory': True} if self.cuda else {}
        super(StudentDataLoader, self).__init__(
            self.data,
            batch_size=batch_size,
            shuffle=self.config.SHUFFLE,
            **additional_options,
        )
