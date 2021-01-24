'''https://keras.io/examples/rl/deep_q_network_breakout/'''
from datetime import datetime
import tensorflow as tf


class Models():
    '''
    Representation of several models, which can be called to generate them.
    Currently dqn and dueling dqn have been implemented
    '''

    @staticmethod
    def dqn(input_dim, num_actions):
        """Return a regular dqn model like in Nature Paper.

        Parameters
        ----------
        input_dim: tuple
            Dimensions of observation space are the input dimension for the dqn
        num_actions: int
            Number of actions in environment are the number of outputs of dqn model
        Returns
        -------
        model: tf.keras.models.Sequential
            The full dqn model
        """
        model = tf.keras.models.Sequential([
            tf.keras.Input(shape=input_dim),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=8, strides=4, activation="relu"),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=4, strides=2, activation="relu"),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=1, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(num_actions),
        ])

        return model

    @staticmethod
    def dueling_dqn(input_dim, num_actions):
        """Return a dueling dqn model like in Nature Paper.

        Parameters
        ----------
        input_dim: tuple
            Dimensions of observation space are the input dimension for the dqn
        num_actions: int
            Number of actions in environment are the number of outputs of dqn model
        Returns
        -------
        model: tf.keras.models.Sequential
            The full dqn model
        """

        inputs = tf.keras.Input(shape=input_dim)
        conv1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=8, strides=4, activation="relu")(inputs)
        conv2 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=4, strides=2, activation="relu")(conv1)
        conv3 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=1, activation="relu")(conv2)
        flatt = tf.keras.layers.Flatten()(conv3)
        dense1 = tf.keras.layers.Dense(512, activation="relu")(flatt)
        dense_v = tf.keras.layers.Dense(1)(dense1)
        dense_a = tf.keras.layers.Dense(num_actions)(dense1)
        output_q = tf.add(dense_v, tf.subtract(
            dense_a, tf.reduce_mean(dense_a, axis=1, keepdims=True)))
        model = tf.keras.models.Model(inputs=inputs, outputs=output_q)

        return model

    @staticmethod
    def save_model(model, path):
        """Saves a model to a given location

        Parameters
        ----------
        model: tf.keras.models.Sequential
            The model to save
        path: String
            The path where to save the model
        """
        now = datetime.now()

        dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
        model.save((path+'model-'+dt_string+".h5").replace(" ", ""))

    @staticmethod
    def load_model(model_path):
        """Loads a model from a given location

        Parameters
        ----------
        model_path: String
            The path where to load the model from
        Returns
        -------
        model: keras Model
            The saved model at location model_path
        """
        return tf.keras.models.load_model(model_path)
