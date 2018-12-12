import os
import time
import torch
import numpy as np
from config import Config
from helper import Helper


class Trainer(object):
    config = Config.instance()
    helper = Helper()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = '../data/checkpoints'

    def __init__(self, model, data_loader):
        self.data_loader = data_loader
        self.model = model
        self.checkpoint_file = model.__class__.__name__ + '_' + str(int(time.time())) + '.pt'
        self.optimizer = torch.optim.SGD(
            params=model.parameters(),
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

    def evaluate(self):
        print("Start evaluation...")
        checkpoint_info = self.load_checkpoint()
        if checkpoint_info is False:
            return
        epoch = checkpoint_info['epoch']
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
                accuracies = np.array(self._get_accuracies(target, output, strict_mode=True))
                accuracies = list(map(np.sum, np.transpose(accuracies)))
                accuracy_sums = np.add(accuracy_sums, accuracies)
        mean_accuracies = list(float('%.3f' % (accuracy_sum / len(self.data_loader.data))) for accuracy_sum in accuracy_sums)
        print("Evaluation is done!")
        print("Total Accuracies:", mean_accuracies)

    def _get_accuracies(self, target, output, strict_mode):
        accuracies = []
        errors_list = list(map(lambda val1, val2: abs(val1 - val2), target, output))
        for errors in errors_list:
            if strict_mode:
                accuracies.append(list((float(abs(error) < 1) for error in errors)))
            else:
                max_output = 20
                accuracies.append(list((max_output - error) / max_output for error in errors)[0])
        return accuracies
