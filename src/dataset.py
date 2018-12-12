import os
import torch
import torch.utils.data
from config import Config
from helper import Helper


class StudentData(torch.utils.data.Dataset):
    config = Config.instance()
    helper = Helper()
    data_map = [
        {'GP': 1, 'MS': 2},
        {'F': 1, 'M': 2},
        'numeric',
        {'U': 1, 'R': 2},
        {'LE3': 1, 'GT3': 2},
        {'T': 1, 'A': 2},
        'numeric',
        'numeric',
        {'teacher': 1, 'health': 2, 'services': 3, 'at_home': 4, 'other': 5},
        {'teacher': 1, 'health': 2, 'services': 3, 'at_home': 4, 'other': 5},
        {'home': 1, 'reputation': 2, 'course': 3, 'other': 4},
        {'mother': 1, 'father': 2, 'other': 3},
        'numeric',
        'numeric',
        'numeric',
        {'yes': 1, 'no': 0},
        {'yes': 1, 'no': 0},
        {'yes': 1, 'no': 0},
        {'yes': 1, 'no': 0},
        {'yes': 1, 'no': 0},
        {'yes': 1, 'no': 0},
        {'yes': 1, 'no': 0},
        {'yes': 1, 'no': 0},
        'numeric',
        'numeric',
        'numeric',
        'numeric',
        'numeric',
        'numeric',
        'numeric',
    ]

    def __init__(self, data_root, subject, is_training_mode):
        self.data_root = os.path.expanduser(data_root)

        if not self.helper.check_path_exists(self.data_root):
            print(self.helper.check_path_exists(self.data_root))
            print(
                "[!] There is no data path! Download data from "
                "`https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip`"
            )
            # self.download(https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip)
            return

        self.loaded_data_path = os.path.join(self.data_root, 'loaded')
        if subject is 'mat':
            self.csv_data_file = os.path.join(self.data_root, 'student-mat.csv')
            self.test_set_size = self.config.MAT_TEST_SET_SIZE
            self.training_file = os.path.join(self.loaded_data_path, 'trianing-mat.pt')
            self.test_file = os.path.join(self.loaded_data_path, 'test-mat.pt')
        elif subject is 'por':
            self.csv_data_file = os.path.join(self.data_root, 'student-por.csv')
            self.test_set_size = self.config.POR_TEST_SET_SIZE
            self.training_file = os.path.join(self.loaded_data_path, 'trianing-por.pt')
            self.test_file = os.path.join(self.loaded_data_path, 'test-por.pt')
        else:
            print("[!] Subject must be `mat` or `por`. There is no option `" + subject + "`")
            return

        if not self.helper.check_path_exists(self.loaded_data_path):
            self.preprocess()

        if is_training_mode:
            self.loaded_data = torch.load(self.training_file)
        else:
            self.loaded_data = torch.load(self.test_file)

    def preprocess(self):
        print("Start preprocessing...")

        preprocessed = {
            'training': {
                'attributes': [],
                'targets': [],
            },
            'test': {
                'attributes': [],
                'targets': [],
            },
        }
        with open(self.csv_data_file, 'r') as data:
            mode = 'test'
            for idx, student in enumerate(data):
                if idx == 0:
                    continue
                if idx == self.test_set_size + 1:
                    mode = 'training'
                student = self.helper.chomp_new_line(student)
                student = student.replace('"', '')
                attributes = student.split(';')
                targets = attributes[-3:]
                attributes = attributes[:-3]
                for idx, attribute in enumerate(attributes):
                    map = self.data_map[idx]
                    if map == 'numeric':
                        attributes[idx] = int(attribute)
                    else:
                        attributes[idx] = map[attribute]
                for idx, target in enumerate(targets):
                    targets[idx] = int(target)
                preprocessed[mode]['attributes'].append(torch.Tensor(attributes))
                preprocessed[mode]['targets'].append(torch.Tensor(targets))

        try:
            os.mkdir(self.loaded_data_path)
        except FileExistsError:
            pass
        with open(self.training_file, 'wb') as f:
            torch.save(preprocessed['training'], f)
        with open(self.test_file, 'wb') as f:
            torch.save(preprocessed['test'], f)

        print("Preprocessing is done!")

    def __getitem__(self, index):
        return self.loaded_data['attributes'][index], self.loaded_data['targets'][index]

    def __len__(self):
        return len(self.loaded_data['attributes'])
