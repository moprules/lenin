import numpy as np
from . import mnet


def mae(preds: np.ndarray, actuals: np.ndarray):
    """
    Расчитываем абсолютную ошибку
    """
    return np.mean(np.abs(preds-actuals))


def rmse(preds: np.ndarray, actuals: np.ndarray):
    """
    Расчитываем абсолютную ошибку
    """
    return np.sqrt(np.mean(np.power(preds-actuals, 2)))


def eval_regression_model(model: mnet.NeuralNetwork,
                          X_test: np.ndarray,
                          y_test: np.ndarray):
    '''
    Compute mae and rmse for a neural network.
    '''
    preds = model.forward(X_test)
    preds = preds.reshape(-1, 1)
    print("\nMean absolute error: {:.2f}".format(mae(preds, y_test)))
    print("Root mean squared error {:.2f}".format(rmse(preds, y_test)))


def calc_accuracy_model(model: mnet.NeuralNetwork, X_test: np.ndarray, y_test: np.ndarray):
    accuracy = np.equal(
        np.argmax(model.predict(X_test), axis=1), y_test).sum()
    accuracy *= 100.0
    accuracy /= X_test.shape[0]
    return f"The model validation accuracy is: {accuracy:.2f}%"


def accuracy_preds(preds_test: np.ndarray, y_test: np.ndarray):
    preds = np.argmax(preds_test, axis=1)
    y = np.argmax(y_test, axis=1)
    accuracy = np.equal(preds, y).sum()
    accuracy *= 100.0
    accuracy /= preds_test.shape[0]
    return accuracy
