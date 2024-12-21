import numpy as np
from . import dataext


class Loss:
    '''
    Потери нейронной сети
    '''

    def __init__(self):
        '''
        Пока ничего не делаем
        '''
        pass

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        '''
        Вычисление значения потерь
        '''
        dataext.assert_same_shape(prediction, target)
        self.prediction = prediction
        self.target = target

        loss_value = self._output()
        return loss_value

    def backward(self) -> np.ndarray:
        '''
        Вычисление градиента потерь по входам функции потерь
        '''
        self.input_grad = self._input_grad()
        dataext.assert_same_shape(self.prediction, self.input_grad)
        return self.input_grad

    def _output(self) -> float:
        '''
        Функция _output должна реализовываться всеми подклассами класса Loss
        '''
        raise NotImplementedError()

    def _input_grad(self) -> float:
        '''
        Функция _output_grad должна реализовываться всеми подклассами класса Loss
        '''
        raise NotImplementedError()


class MeanSquaredError(Loss):

    def __init__(self, normalize: bool = False) -> None:
        super().__init__()
        self.normalize = normalize

    def _output(self) -> float:

        if self.normalize:
            self.prediction = (self.prediction /
                               self.prediction.sum(axis=1, keepdims=True))

        loss = (np.sum(np.power(self.prediction - self.target, 2)) /
                self.prediction.shape[0])

        return loss

    def _input_grad(self) -> np.ndarray:

        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]


class SoftmaxCrossEntropy(Loss):
    def __init__(self, eps: float = 1e-9) -> None:
        super().__init__()
        self.eps = eps
        self.single_class = False

    def _output(self) -> float:

        # if the network is just outputting probabilities
        # of just belonging to one class:
        if self.target.shape[1] == 0:
            self.single_class = True

        # if "single_class", apply the "normalize" operation defined above:
        if self.single_class:
            self.prediction = dataext.normalize(self.prediction)
            self.target = dataext.normalize(elf.target)

        # applying the softmax function to each row (observation)
        softmax_preds = dataext.softmax(self.prediction, axis=1)

        # clipping the softmax output to prevent numeric instability
        self.softmax_preds = np.clip(softmax_preds, self.eps, 1 - self.eps)

        # actual loss computation
        softmax_cross_entropy_loss = (-1.0 * self.target * np.log(self.softmax_preds)
                                      - (1.0 - self.target) * np.log(1 - self.softmax_preds))

        return np.sum(softmax_cross_entropy_loss) / self.prediction.shape[0]

    def _input_grad(self) -> np.ndarray:

        # if "single_class", "un-normalize" probabilities before returning gradient:
        if self.single_class:
            return dataext.unnormalize(self.softmax_preds - self.target)
        else:
            return (self.softmax_preds - self.target) / self.prediction.shape[0]
