import numpy as np
from . import mnet


class Optimizer:
    '''
    Базовый класс оптимизатора нейросети
    '''

    def __init__(self,
                 lr: float = 0.01,
                 final_lr: float = 0,
                 decay_type: str = None):
        '''У оптимизатора должна быть начальная скорость обучения'''
        self.lr = lr
        self.final_lr = final_lr
        self.decay_type = decay_type
        self.first = True

    def _setup_decay(self, max_epochs: int) -> None:
        if not self.decay_type:
            return
        elif self.decay_type == "exponential":
            self.decay_per_epoch = np.power(self.final_lr / self.lr, 1.0 / (max_epochs - 1))
        elif self.decay_type == "linear":
            self.decay_per_epoch = (self.lr - self.final_lr) / (max_epochs - 1)

    def _decay_lr(self) -> None:
        if not self.decay_type:
            return
        elif self.decay_type == "exponential":
            self.lr *= self.decay_per_epoch
        elif self.decay_type == "linear":
            self.lr -= self.decay_per_epoch

    def step(self, net: mnet.NeuralNetwork) -> None:

        for param, param_grad in zip(net.params(), net.param_grads()):
            self._update_rule(param=param, grad=param_grad)

    def _update_rule(self, **kwargs) -> None:
        raise NotImplementedError()


class SGD(Optimizer):
    '''
    Стохастический градиентный оптимизатор
    '''

    def __init__(self,
                 lr: float = 0.01,
                 final_lr: float = 0,
                 decay_type: str = None):
        '''Пока ничего'''
        super().__init__(lr, final_lr, decay_type)

    def _update_rule(self, **kwargs) -> None:
        update = self.lr * kwargs["grad"]
        kwargs["param"] -= update


class SGDMomentum(Optimizer):
    def __init__(self,
                 lr: float = 0.01,
                 final_lr: float = 0,
                 decay_type: str = None,
                 momentum: float = 0.9):
        super().__init__(lr, final_lr, decay_type)
        self.momentum = momentum

    def step(self, net: mnet.NeuralNetwork) -> None:
        if self.first:
            # Для первой итерации все предыдущие скорости = 0
            self.velocities = [np.zeros_like(param) for param in net.params()]
            self.first = False

        for param, param_grad, velocity in zip(net.params(), net.param_grads(), self.velocities):
            self._update_rule(param=param, grad=param_grad, velocity=velocity)

    def _update_rule(self, **kwargs) -> None:
        """Обновление по скорости и инерции"""
        # Обновляем скорость
        kwargs["velocity"] *= self.momentum
        kwargs["velocity"] += self.lr * kwargs["grad"]
        # Обновляем параметры
        kwargs["param"] -= kwargs["velocity"]
