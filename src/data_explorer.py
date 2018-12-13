import os
import time
import math
import torch
import numpy as np
from config import Config
from data_loader import StudentDataLoader
from model import NNModel
from helper import Helper


class DataExplorer(object):
    config = Config.instance()
    helper = Helper()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = os.path.join('..', 'data', 'checkpoints')

    def __init__(self, training_mode):
        self.data_loader = StudentDataLoader(
            batch_size=self.config.BATCH_SIZE,
            subject=self.config.SUBJECT,
            training_mode = training_mode,
        )
        self.model = NNModel(training_mode)
        self.checkpoint_file = self.model.__class__.__name__ + '_' + str(int(time.time())) + '.pt'
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            # weight_decay=self.config.WEIGHT_DECAY,
        )
        self.criterion = torch.nn.MSELoss()
        try:
            os.mkdir(self.checkpoint_path)
        except FileExistsError:
            pass

    def save_checkpoint(self, checkpoint_info):
        """ Save meta information of training as a checkpoint
        :param checkpoint_info: Should include `epoch`, `model_state`, `optimizer_state`.
        """
        torch.save(checkpoint_info, os.path.join(self.checkpoint_path, self.checkpoint_file))

    def load_checkpoint(self):
        root, _, files = list(os.walk(self.checkpoint_path))[0]
        if len(files) == 0:
            print("[!] There is no checkpoints.")
            return False
        loaded_file = files[-1]
        print("[+] Loaded a checkpoint:", loaded_file)
        return torch.load(os.path.join(root, loaded_file))


class Trainer(DataExplorer):
    def __init__(self):
        super(__class__, self).__init__(True)

    def train(self, max_epoch, accuracy_print_frequency=50):
        print("Start training...")
        for epoch in range(max_epoch):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(device=self.device), target.to(device=self.device)
                self.optimizer.zero_grad()  # Initialize gradients
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
            if epoch % accuracy_print_frequency == 0:
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                })
                print("Training is ongoing... ({} / {})".format(epoch, max_epoch))
        print("Training is done!")


class Evaluator(DataExplorer):
    def __init__(self):
        super(__class__, self).__init__(False)

    def evaluate(self):
        print("Start evaluation...")
        checkpoint_info = self.load_checkpoint()
        if checkpoint_info is False:
            return
        optimizer_state = checkpoint_info['optimizer_state']
        model_state = checkpoint_info['model_state']
        self.optimizer.load_state_dict(optimizer_state)
        self.model.load_state_dict(model_state)

        accuracy_sums = [0] * self.config.OUTPUT_SIZE
        with torch.no_grad():
            for data, target in self.data_loader:
                data, target = data.to(device=self.device), target.to(device=self.device)
                self.optimizer.zero_grad()  # Initialize gradients
                output = self.model(data)
                accuracies = np.array(self._get_accuracies(target, output, strict_mode=False))
                accuracies = list(map(np.sum, np.transpose(accuracies)))
                accuracy_sums = np.add(accuracy_sums, accuracies)

        rmse = self._get_rmse_list(target, output)
        mean_accuracies = list(
            float('%.3f' % (accuracy_sum / len(self.data_loader.data))) for accuracy_sum in accuracy_sums
        )
        print("Evaluation is done!")
        print("Total Accuracies:", mean_accuracies)
        print("RMSE:", rmse)

    def _get_rmse_list(self, target, output):
        print(type(output))
        output = torch.t(output)
        target = torch.t(target)
        output_size = self.config.OUTPUT_SIZE
        rmse_list = []
        for i in range(output_size):
            rmse_list.append(float('%.3f' % math.sqrt(self.criterion(output[i], target[i]))))
        return rmse_list

    def _get_accuracies(self, target, output, strict_mode):
        accuracies = []
        errors_list = list(map(lambda val1, val2: abs(val1 - val2), target, output))
        for errors in errors_list:
            if strict_mode:
                accuracies.append(list((float(abs(error) < 1) for error in errors)))
            else:
                max_output = 20
                accuracies.append(list((1 - error / max_output) for error in errors))
        return accuracies
