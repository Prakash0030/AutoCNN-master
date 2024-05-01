import tensorflow as tf

class Prediction:
    def __init__(self):
        self.policy_model=self.policy_network()
        self.value_model=self.value_network()
    def policy_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(256,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='softmax')  # Assuming 32 possible moves
        ])
        return model
    
    def value_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(256,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Output a value between 0 and 1
        ])
        return model
    
    def predict_move(self, features):
        policy_output=self.policy_model.predict(features.reshape(1, -1))
        value_output=self.value_model.predict(features.reshape(1, -1))
        return policy_output, value_output
