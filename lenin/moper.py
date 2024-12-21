import numpy as np
from . import dataext


class Operation:
    '''
    Базовый класс операции нейросети
    '''

    def __init__(self):
        pass

    def forward(self, input_: np.ndarray) -> np.ndarray:
        '''
        Хранение ввода в атрибуте экземпляра self.input_
        Вызов функции self._output()
        '''
        self.input_ = input_
        self.output = self._output()
        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Вызов функции self._input_grad()
        Проверка совпадения размерностей
        '''

        dataext.assert_same_shape(self.output, output_grad)
        self.input_grad = self._input_grad(output_grad)
        dataext.assert_same_shape(self.input_, self.input_grad)
        return self.input_grad

    def _output(self) -> np.ndarray:
        '''
        Метод _output определяется для каждой операции
        '''
        raise NotImplementedError()

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Метод _input_grad определяется для каждой операции
        '''
        raise NotImplementedError()


class ParamOperation(Operation):
    '''
    Операция с параметрами
    '''

    def __init__(self, param: np.ndarray) -> np.ndarray:
        '''
        Метод ParamOperation
        '''
        super().__init__()
        self.param = param

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Вызов self._input_grad и self._param_grad
        Проверка размерностей
        '''

        dataext.assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        dataext.assert_same_shape(self.input_, self.input_grad)
        dataext.assert_same_shape(self.param, self.param_grad)

        return self.input_grad

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Во всех подклассах ParamOperation должна быть реализация
        метода _param_grad
        '''

        raise NotImplementedError()


class WeightMultiply(ParamOperation):
    '''
    Умножение весов в нейронной сети
    '''

    def __init__(self, W: np.ndarray):
        '''
        Инициализация класса Operation c self.param = W
        '''
        super().__init__(W)

    def _output(self) -> np.ndarray:
        '''
        Вычисление выхода
        '''
        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Вычисление входного градиента
        '''
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Вычисление градиента параметров
        '''
        return np.dot(np.transpose(self.input_, (1, 0)), output_grad)


class BiasAdd(ParamOperation):
    '''
    Прибавление отклонений
    '''

    def __init__(self, B: np.ndarray):
        '''
        Инициализация класса Operation с self.param = B
        Проверка размерностей
        '''

        assert B.shape[0] == 1
        super().__init__(B)

    def _output(self) -> np.ndarray:
        '''
        Вычисление выхода
        '''
        return self.input_ + self.param

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Вычисление входного градиента
        '''
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Вычисление градиента параметров
        '''
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])


class Linear(Operation):
    """
    Linear activation function
    """

    def __init__(self) -> None:
        super().__init__()

    def _output(self) -> np.ndarray:
        return self.input_

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad


class Sigmoid(Operation):
    '''
    Сигмоидная функция активации
    '''

    def __init__(self):
        '''
        Пока ничего не делаем
        '''
        super().__init__()

    def _output(self) -> np.ndarray:
        '''
        Вычисление выхода
        '''
        return 1.0 / (1.0+np.exp(-1.0*self.input_))

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Вычисление входного градиента
        '''
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad


class Tanh(Operation):
    """
    Hyperbolic tangent activation function
    """

    def __init__(self) -> None:
        super().__init__()

    def _output(self) -> np.ndarray:
        return np.tanh(self.input_)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:

        return output_grad * (1 - self.output * self.output)


class ReLU(Operation):
    """
    Hyperbolic tangent activation function
    """

    def __init__(self) -> None:
        super().__init__()

    def _output(self) -> np.ndarray:
        return np.clip(self.input_, 0, None)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:

        mask = self.output >= 0
        return output_grad * mask


class Dropout(Operation):
    def __init__(self,
                 keep_prob: float = 0.8):
        super().__init__()
        self.keep_prob = keep_prob

    def forward(self, input_: np.ndarray, inference: bool = False) -> np.ndarray:
        '''
        Хранение ввода в атрибуте экземпляра self.input_
        Вызов функции self._output()
        '''
        self.input_ = input_
        self.output = self._output(inference=inference)
        return self.output

    def _output(self, inference: bool = False) -> np.ndarray:
        if inference:
            return self.input_ * self.keep_prob
        else:
            self.mask = np.random.binomial(1, self.keep_prob,
                                           size=self.input_.shape)
            return self.input_ * self.mask

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad * self.mask


class Conv2DOpBase(ParamOperation):

    def __init__(self, W: np.ndarray):
        super().__init__(W)
        self.param_size = W.shape[2]
        self.param_pad = self.param_size // 2

    def _pad_1d(self, inp: np.ndarray) -> np.ndarray:
        z = np.array([0])
        z = np.repeat(z, self.param_pad)
        return np.concatenate([z, inp, z])

    def _pad_1d_batch(self, inp: np.ndarray) -> np.ndarray:
        outs = [self._pad_1d(obs) for obs in inp]
        return np.stack(outs)

    def _pad_2d_obs(self, inp: np.ndarray):
        """
        Input is a 2 dimensional, square, 2D Tensor
        """
        inp_pad = self._pad_1d_batch(inp)

        other = np.zeros((self.param_pad, inp.shape[0] + self.param_pad * 2))

        return np.concatenate((other, inp_pad, other))

    def _pad_2d_channel(self, inp: np.ndarray):
        """
        inp has dimension [num_channels, img_width, img_height]
        """
        return np.stack([self._pad_2d_obs(channel) for channel in inp])

    def _pad_conv(self, inp: np.ndarray):
        '''
        inp: [batch_size, num_channels, img_width, img_height]
        obs: [num_channels, img_width, img_height]
        '''
        return np.stack([self._pad_2d_channel(obs) for obs in inp])


class Conv2DOpSlow(Conv2DOpBase):

    def _compute_output_obs(self, obs: np.ndarray) -> np.ndarray:
        '''
        Вычисляем свёртку для одного наблюдения obs из набора batch
        obs: [channels, img_width, img_height]
        self.param: [in_channels, out_channels, param_width, param_height]
        '''
        obs_pad = self._pad_2d_channel(obs)

        in_channels = self.param.shape[0]
        out_channels = self.param.shape[1]
        img_width, img_height = obs.shape[1:]

        out = np.zeros((out_channels, img_width, img_height))

        for c_in in range(in_channels):
            for c_out in range(out_channels):
                for o_w in range(img_width):
                    for o_h in range(img_height):
                        for p_w in range(self.param_size):
                            for p_h in range(self.param_size):
                                out[c_out][o_w][o_h] += (self.param[c_in][c_out][p_w][p_h]
                                                         * obs_pad[c_in][o_w+p_w][o_h+p_h])
        return out

    def _output(self) -> np.ndarray:
        '''
        Вычисляем свёртку для набора измерений
        self.input_: [batch_size, in_channels, img_width, img_height]
        obs: [channels, img_width, img_height]
        self.param: [in_channels, out_channels, param_width, param_height]
        '''
        outs = [self._compute_output_obs(obs) for obs in self.input_]
        return np.stack(outs)

    def _compute_grad_obs(self,
                          inp_obs: np.ndarray,
                          out_grad_obs: np.ndarray) -> np.ndarray:
        '''
        Вычисляем входной градиент для одного наблюдения obs из набора batch
        inp_obs: [in_channels, img_width, img_height]
        out_grad_obs: [out_channels, img_width, img_height]
        self.param: [in_channels, out_channels, img_width, img_height]
        '''
        inp_grad_obs = np.zeros_like(inp_obs)

        img_width, img_height = inp_obs.shape[1:]

        in_channels = self.param.shape[0]
        out_channels = self.param.shape[1]
        out_grad_obs_pad = self._pad_2d_channel(out_grad_obs)

        for c_in in range(in_channels):
            for c_out in range(out_channels):
                for i_w in range(img_width):
                    for i_h in range(img_height):
                        for p_w in range(self.param_size):
                            for p_h in range(self.param_size):
                                inp_grad_obs[c_in][i_w][i_h] += (out_grad_obs_pad[c_out][i_w+self.param_size-p_w-1][i_h+self.param_size-p_h-1]
                                                                 * self.param[c_in][c_out][p_w][p_h])

        return inp_grad_obs

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Вычисляем входной градиент для набора измерений
        output_grad: [batch_size, out_channels, img_width, img_height]
        self.input_: [batch_size, in_channels, img_width, img_height] 
        '''
        grads = [self._compute_grad_obs(self.input_[i], output_grad[i])
                 for i in range(output_grad.shape[0])]
        return np.stack(grads)

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Вычисляем градиент параметра
        output_grad: [batch_size, out_channels, img_width, img_height]
        self.input_: [batch_size, in_channels, img_width, img_height]
        self.param: [in_channels, out_channels, img_width, img_height]
        '''
        param_grad = np.zeros_like(self.param)

        in_channels = self.param.shape[0]
        out_channels = self.param.shape[1]
        inp_pad = self._pad_conv(self.input_)
        img_width, img_height = output_grad.shape[2:]
        batch_size = output_grad.shape[0]

        for i in range(batch_size):
            for c_in in range(in_channels):
                for c_out in range(out_channels):
                    for o_w in range(img_width):
                        for o_h in range(img_height):
                            for p_w in range(self.param_size):
                                for p_h in range(self.param_size):
                                    param_grad[c_in][c_out][p_w][p_h] += (inp_pad[i][c_in][o_w+p_w][o_h+p_h]
                                                                          * output_grad[i][c_out][o_w][o_h])
        return param_grad


class Conv2DOpFast(Conv2DOpBase):

    def _get_image_patches(self, input_: np.ndarray):
        imgs_batch_pad = self._pad_conv(input_)
        patches = []
        img_height = imgs_batch_pad.shape[2]
        for h in range(img_height - self.param_size + 1):
            for w in range(img_height - self.param_size + 1):
                patch = imgs_batch_pad[:, :, h: h +
                                       self.param_size, w: w + self.param_size]
                patches.append(patch)
        return np.stack(patches)

    def _output(self):
        """
        self.input_: [batch_size, channels, img_width, img_height]
        self.param: [in_channels, out_channels, fil_width, fil_height]
        """
        batch_size = self.input_.shape[0]
        img_height = self.input_.shape[2]
        img_size = self.input_.shape[2] * self.input_.shape[3]
        patch_size = (self.param.shape[0]
                      * self.param.shape[2]
                      * self.param.shape[3])

        patches = self._get_image_patches(self.input_)

        patches_reshaped = (
            patches
            .transpose(1, 0, 2, 3, 4)
            .reshape(batch_size, img_size, -1)
        )

        param_reshaped = (
            self.param
            .transpose(0, 2, 3, 1)
            .reshape(patch_size, -1)
        )

        output_reshaped = (
            np.matmul(patches_reshaped, param_reshaped)
            .reshape(batch_size, img_height, img_height, -1)
            .transpose(0, 3, 1, 2)
        )

        return output_reshaped

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:

        batch_size = self.input_.shape[0]
        img_size = self.input_.shape[2] * self.input_.shape[3]
        img_height = self.input_.shape[2]

        output_patches = (
            self._get_image_patches(output_grad)
            .transpose(1, 0, 2, 3, 4)
            .reshape(batch_size * img_size, -1)
        )

        param_reshaped = (
            self.param
            .reshape(self.param.shape[0], -1)
            .transpose(1, 0)
        )

        return (
            np.matmul(output_patches, param_reshaped)
            .reshape(batch_size, img_height, img_height, self.param.shape[0])
            .transpose(0, 3, 1, 2)
        )

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:

        batch_size = self.input_.shape[0]
        img_size = self.input_.shape[2] * self.input_.shape[3]
        in_channels = self.param.shape[0]
        out_channels = self.param.shape[1]

        in_patches_reshape = (
            self._get_image_patches(self.input_)
            .reshape(batch_size * img_size, -1)
            .transpose(1, 0)
        )

        out_grad_reshape = (
            output_grad
            .transpose(0, 2, 3, 1)
            .reshape(batch_size * img_size, -1)
        )

        return (
            np.matmul(in_patches_reshape, out_grad_reshape)
            .reshape(in_channels, self.param_size, self.param_size, out_channels)
            .transpose(0, 3, 1, 2)
        )


class Flatten(Operation):
    def __init__(self):
        super().__init__()

    def _output(self) -> np.ndarray:
        return self.input_.reshape(self.input_.shape[0], -1)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad.reshape(self.input_.shape)
