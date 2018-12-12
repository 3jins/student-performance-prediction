import torch
from config import Config
from helper import Helper


class Trainer(object):
    config = Config.instance()
    helper = Helper()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, model, data_loader):
        self.data_loader = data_loader
        self.model = model
        self.prefix = model.__class__.__name__ + "_"
        self.optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=self.config.LEARNING_RATE,
            # weight_decay=self.config.WEIGHT_DECAY,
        )
        self.criterion = torch.nn.MSELoss()

    def train(self, max_epoch, accuracy_print_frequency=10):
        print("Start training...")
        accuracy_sums = [0] * self.config.OUTPUT_SIZE
        accuracy_sum_cnt = 0
        for epoch in range(max_epoch):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(device=self.device), target.to(device=self.device)
                self.optimizer.zero_grad()  # Initialize gradients
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                accuracy_sums = [
                    x + y for x, y in zip(accuracy_sums, self._get_accuracies(target, output, strict_mode=True))
                ]
                accuracy_sum_cnt += 1
            if epoch % accuracy_print_frequency == 0:
                self._print_accuracy(accuracy_sums, epoch, accuracy_sum_cnt)
                accuracy_sums = [0] * self.config.OUTPUT_SIZE
                accuracy_sum_cnt = 0
            if epoch == max_epoch - 1:
                print("{0} [Last Result] {0}".format("=" * 12))
                print(output)
                print("=" * 40)
        print("Training is done!")

    def evaluate(self):
        raise NotImplementedError

    def _get_accuracies(self, target, output, strict_mode):
        errors = list(map(lambda val1, val2: abs(val1 - val2), target, output))
        if strict_mode:
            return list((float(abs(error) < 1) for error in errors[0]))
        else:
            max_output = 20
            return list((max_output - error) / max_output for error in errors)[0]

    # TODO(3jin): Make it to adapt to the output size
    def _print_accuracy(self, accuracy_sums, epoch, accuracy_sum_cnt):
        print("epoch[{}] - accuracies: [{}, {}, {}]".format(
            epoch,
            float('%.2f' % (accuracy_sums[0] / accuracy_sum_cnt)),
            float('%.2f' % (accuracy_sums[1] / accuracy_sum_cnt)),
            float('%.2f' % (accuracy_sums[2] / accuracy_sum_cnt))
        ))
