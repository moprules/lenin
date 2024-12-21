import numpy as np
import typing
from . import moper
from . import dataext


class Layer:
    '''
    Слой нейроннов в нейросети
    '''

    def __init__(self, neurons: int, dropout: float = 1.0, seed: float = None):
        '''
        Число нейронов примерно соответствует ширине слоя
        '''
        self.neurons = neurons
        self.dropout = moper.Dropout(keep_prob=dropout)
        self._seed = seed
        self.first = True
        self.params: typing.List[np.ndarray] = []
        self.param_grads: typing.List[np.ndarray] = []
        self.operations: typing.List[moper.Operation] = []

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value: float):
        self._seed = value

    def _setup_layer(self, input_: np.ndarray) -> None:
        '''
        Функция _setup_layer реализуется в каждом слое
        '''
        raise NotImplementedError()

    def forward(self, input_: np.ndarray, inference: bool = False) -> np.ndarray:
        '''
        Передача входа вперёд через серию операций
        '''
        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for operation in self.operations:
            input_ = operation.forward(input_)

        if self.dropout.keep_prob < 1.0:
            input_ = self.dropout.forward(input_=input_, inference=inference)

        self.output = input_
        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Передача output_grad назад через серию операций
        Проверка размерностей
        '''
        dataext.assert_same_shape(self.output, output_grad)

        if self.dropout.keep_prob < 1.0:
            output_grad = self.dropout.backward(output_grad)

        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)

        input_grad = output_grad

        self._param_grads()
        return input_grad

    def _param_grads(self) -> None:
        '''
        Извлечение _param_grads из операций слоя
        '''
        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, moper.ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> np.ndarray:
        '''
        Извлечение _params из операций слоя
        '''
        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, moper.ParamOperation):
                self.params.append(operation.param)


class Dense(Layer):
    '''
    Полносвязный слой, наследующийся от Layer
    '''

    def __init__(self,
                 neurons: int,
                 dropout: float = 1.0,
                 activation: moper.Operation = moper.Linear(),
                 weight_init: str = "standard",
                 seed=None) -> None:
        '''
        Для инициализации нужна функция активации
        '''
        super().__init__(neurons, dropout, seed)
        self.activation = activation
        self.weight_init = weight_init

    def _setup_layer(self, input_: np.ndarray) -> None:
        '''
        Определение операций для полносвязного слоя
        '''

        if self.seed is not None:
            np.random.seed(self.seed)

        num_in = input_.shape[1]

        if self.weight_init == "glorot":
            scale = np.sqrt(2 / (num_in + self.neurons))
        else:
            scale = 1.0

        self.params = []

        # веса
        self.params.append(
            np.random.normal(loc=0, scale=scale, size=(num_in, self.neurons))
        )

        # отклонения
        self.params.append(
            np.random.normal(loc=0, scale=scale, size=(1, self.neurons))
        )

        self.operations = [
            moper.WeightMultiply(self.params[0]),
            moper.BiasAdd(self.params[1]),
            self.activation
        ]

        return None


class Conv2D(Layer):
    '''
    Свёрточный слой 2D изображения
    '''

    def __init__(self,
                 out_channels: int,
                 param_size: int,
                 dropout: float = 1.0,
                 activation: moper.Operation = moper.Linear(),
                 weight_init: str = "standard",
                 flatten: bool = False,
                 fast: bool = True,
                 seed=None) -> None:
        '''
        Для инициализации нужна функция активации
        '''
        super().__init__(out_channels, dropout, seed)
        self.out_channels = out_channels
        self.param_size = param_size
        self.activation = activation
        self.weight_init = weight_init
        self.flatten = flatten
        self.fast = fast

    def _setup_layer(self, input_: np.ndarray) -> None:
        '''
        Определение операций для свёрточного слоя
        '''
        self.params = []
        in_channels = input_.shape[1]

        if self.weight_init == "glorot":
            scale = 2 / (in_channels + self.out_channels)
        else:
            scale = 1.0

        conv_param = np.random.normal(
            loc=0,
            scale=scale,
            size=(
                in_channels,
                self.out_channels,
                self.param_size,
                self.param_size
            )
        )

        self.params.append(conv_param)

        self.operations = []
        if self.fast:
            self.operations.append(moper.Conv2DOpFast(conv_param))
        else:
            self.operations.append(moper.Conv2DOpSlow(conv_param))
        self.operations.append(self.activation)

        if self.flatten:
            self.operations.append(moper.Flatten())

        return None
