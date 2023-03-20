import pickle
from typing import Dict, List, Any, Union
import numpy as np
# Keras
import tensorflow as tf
from tensorflow import keras
from keras.utils import pad_sequences


def load_data() -> Dict[str, Union[List[Any], int]]:
    path = "keras-data.pickle"
    with open(file=path, mode="rb") as file:
        data = pickle.load(file)

    return data


def preprocess_data(data: Dict[str, Union[List[Any], int]]) -> Dict[str, Union[List[Any], np.ndarray, int]]:
    """
    Preprocesses the data dictionary. Both the training-data and the test-data must be padded
    to the same length; play around with the maxlen parameter to trade off speed and accuracy.
    """
    maxlen = data["max_length"]//16
    #print("maxlen: ", maxlen)
    data["x_train"] = pad_sequences(data['x_train'], maxlen=maxlen)
    data["y_train"] = np.asarray(data['y_train'])
    data["x_test"] = pad_sequences(data['x_test'], maxlen=maxlen)
    data["y_test"] = np.asarray(data['y_test'])

    return data


def train_model(data: Dict[str, Union[List[Any], np.ndarray, int]], model_type="feedforward") -> float:
    """
    Build a neural network of type model_type and train the model on the data.
    Evaluate the accuracy of the model on test data.

    :param data: The dataset dictionary to train neural network on
    :param model_type: The model to be trained, either "feedforward" for feedforward network
                        or "recurrent" for recurrent network
    :return: The accuracy of the model on test data
    """

    # TODO build the model given model_type, train it on (data["x_train"], data["y_train"])
    #  and evaluate its accuracy on (data["x_test"], data["y_test"]). Return the accuracy
    accuracy = [0, 0]

    # shape_x_train
    # shape_y_train
    # shape_x_test
    # shape_y_test

    print("data x train ", data["x_train"].shape)
    print("data y train ", data["y_train"].shape)
    print("data x test ", data["x_test"].shape)
    print("data y test ", data["y_test"].shape)

    x_train = tf.keras.utils.pad_sequences(
        data["x_train"], maxlen=(data["max_length"]//8))
    y_train = tf.keras.utils.pad_sequences(
        data["y_train"].reshape(data["x_train"].shape[0], 1), maxlen=data["max_length"]//8)
    x_test = tf.keras.utils.pad_sequences(
        data["x_test"], maxlen=data["max_length"]//8)
    y_test = tf.keras.utils.pad_sequences(
        data["y_test"].reshape(data["x_test"].shape[0], 1), maxlen=data["max_length"]//8)

    model = tf.keras.Sequential()

    if model_type == "recurrent":
        print(model_type)

        """
        tf.keras.layers.LSTM
        tf.keras.layers.Dense
        
        """
        model.add(tf.keras.layers.Embedding(
            input_dim=data["vocab_size"], output_dim=32))
        model.add(tf.keras.layers.LSTM(16))
        model.add(tf.keras.layers.Dense(132, "softmax"))

        model.summary()
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.FalseNegatives()])
        model.fit(x_train, y_train, batch_size=128, epochs=1)

    else:
        print("feedforward")

        model.add(tf.keras.layers.Embedding(
            input_dim=data["vocab_size"], output_dim=10))
        model.add(tf.keras.layers.Dense(100, "sigmoid"))
        #model.add(tf.keras.layers.Dense(10, "relu"))
        model.add(tf.keras.layers.Dense(1, "sigmoid"))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[tf.keras.metrics.BinaryAccuracy(),
                               tf.keras.metrics.FalseNegatives()])
        """ model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[tf.keras.metrics.BinaryAccuracy()]) """
        model.fit(x_train, y_train, batch_size=128, epochs=1)

    print("Evaluate on test data")
    accuracy = model.evaluate(x_test, y_test, batch_size=128)
    print("test loss, test acc:", accuracy)

    #model.evaluate(data["x_test"], data["y_test"])

    # model.evaluate(x test, y test)

    # Pad_sequence ?
    #
    # model = tf.keras.Sequential()
    #
    #       tf.keras.layers.Embedding
    #       (model.add(tf.keras.layers.Embedding(<parameters>)))
    #       tf.keras.layers.Dense
    #       tf.keras.layers.LSTM
    #
    # model.fit(epochs=1)
    #
    # model.evaluate(x test, y test)

    return accuracy[1]
    # pass


def main() -> None:
    print("1. Loading data...")
    keras_data = load_data()
    print("2. Preprocessing data...")
    keras_data = preprocess_data(keras_data)
    print("3. Training feedforward neural network...")
    #fnn_test_accuracy = train_model(keras_data, model_type="feedforward")
    # print('Model: Feedforward NN.\n'
    #      f'Test accuracy: {fnn_test_accuracy:.3f}')
    print("4. Training recurrent neural network...")
    rnn_test_accuracy = train_model(keras_data, model_type="recurrent")
    print('Model: Recurrent NN.\n'
          f'Test accuracy: {rnn_test_accuracy:.3f}')


if __name__ == '__main__':
    main()
