import numpy as np
import typing
from copy import deepcopy
from . import mnet
from . import moptim
from . import mmetric
from . import dataext


class Trainer:
    """Обучение нейросети"""

    def __init__(self,
                 net: mnet.NeuralNetwork,
                 optim: moptim.Optimizer):
        """
        Для обучения нужны нейросеть и оптимизатор
        Нейросеть назначается атрибутом экземпляра оптимизатор
        """
        self.net = net
        self.optim = optim
        self.best_loss = float("inf")

    def generate_batches(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         size: int = 32) -> typing.Tuple[np.ndarray]:
        '''
        Generates batches for training 
        '''
        assert X.shape[0] == y.shape[0], '''
        features and target must have the same number of rows, instead
        features has {0} and target has {1}
        '''.format(X.shape[0], y.shape[0])

        N = X.shape[0]

        for ii in range(0, N, size):
            X_batch, y_batch = X[ii:ii+size], y[ii:ii+size]

            yield X_batch, y_batch

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_test: np.ndarray, y_test: np.ndarray,
            epochs: int = 100,
            eval_every: int = 10,
            batch_size: int = 32,
            seed: float = None,
            restat: bool = True,
            early_stopping: bool = True,
            conv_testing: bool = False):
        '''
        Подгонка нейронной сети под обучающие данные за некоторое число эпох
        Через каждые eval_every эпох выполняется оценка
        '''

        self.optim._setup_decay(max_epochs=epochs)

        if seed is not None:
            np.random.seed(seed)

        if restat:
            for layer in self.net.layers:
                layer.first = True
            self.best_loss = float("inf")

        for e in range(epochs):

            X_train, y_train = dataext.permute_data(X_train, y_train)

            batch_generator = self.generate_batches(
                X_train, y_train, batch_size)

            for ii, (X_batch, y_batch) in enumerate(batch_generator):
                self.net.train_batch(X_batch, y_batch)
                self.optim.step(self.net)

                if conv_testing:
                    if ii % 10 == 0:
                        preds_batch = self.net.predict(X_batch)
                        loss_batch = self.net.loss.forward(preds_batch, y_batch)
                        print("batch", ii, "loss_batch", loss_batch)

                    if ii % 100 == 0 and ii > 0:
                        preds_test = self.net.predict(X_test)
                        loss = self.net.loss.forward(preds_test, y_test)
                        accuracy = mmetric.accuracy_preds(preds_test, y_test)
                        print(
                            f"[{ii:3} batch] loss: {loss:5.3f} | accuracy: {accuracy:5.2f}%")

            if (e+1) % eval_every == 0:
                # Каждые eval_every эпох оцениваем ошибку
                preds_test = self.net.predict(X_test)
                loss = self.net.loss.forward(preds_test, y_test)
                accuracy = mmetric.accuracy_preds(preds_test, y_test)
                print(
                    f"[{e+1:3} epoch] loss: {loss:5.3f} | accuracy: {accuracy:5.2f}%")

                if early_stopping:
                    if loss <= self.best_loss:
                        self.best_loss = loss
                        last_model = deepcopy(self.net)
                    else:
                        self.net = last_model
                        print(f"\nLoss increased after epoch: {e+1}")
                        print(
                            f"using model[{e+1-eval_every:3}] loss: {self.best_loss:.3f}")
                        break

            # В конце каждой эпохи
            self.optim._decay_lr()
