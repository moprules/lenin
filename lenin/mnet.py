import numpy as np
import typing
from . import mlayer
from . import mloss


class NeuralNetwork:
    '''
    Класс нейронной сети
    '''

    def __init__(self, layers: typing.List[mlayer.Layer],
                 loss: mloss.Loss,
                 seed: float = None):
        '''
        Нейросети нужны слои и потери
        '''
        self.layers = layers
        self.loss = loss
        self.seed = seed

        if seed is not None:
            for layer in self.layers:
                layer.seed = self.seed

    def forward(self, x_batch: np.ndarray, inference: bool = False) -> np.ndarray:
        '''
        Передача данных через последовательность слоёв
        '''
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(input_=x_out, inference=inference)
        return x_out

    def predict(self, x_batch: np.ndarray):
        """Предсказания нейросети без зануления весов"""
        return self.forward(x_batch=x_batch, inference=True)

    def backward(self, loss_grads: np.ndarray) -> None:
        '''
        Передача данных назад через последовательность слоёв
        '''
        grad = loss_grads
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return None

    def train_batch(self,
                    x_batch: np.ndarray,
                    y_batch: np.ndarray) -> float:
        '''
        Передача данных через последовательность слоёв
        Вычисление потерь
        Передача данных назад через последовательность слоёв
        '''
        # При тренировки нейросети используем dropout
        predictions = self.forward(x_batch, inference=False)
        loss = self.loss.forward(predictions, y_batch)
        self.backward(self.loss.backward())
        return loss

    def params(self):
        '''
        Получение параметров нейронной сети
        '''
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        '''
        Получение градиента потерь по отношению к параметрам нейросети
        '''
        for layer in self.layers:
            yield from layer.param_grads
