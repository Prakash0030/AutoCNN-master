import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from auto_cnn import Prediction

tf.get_logger().setLevel('INFO')

from auto_cnn.gan import AutoCNN

import random

random.seed(42)
tf.random.set_seed(42)

# Function to parse tfrecord file
def parse_function(p):
    # Define your tfrecord again. It might be something like this
    keys_to_features = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string, default_value='jpeg'),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
    }

    # Load one example
    parsed_features = tf.io.parse_single_example(p, keys_to_features)

    # Turn your image string into an array
    parsed_features['image']=tf.io.decode_jpeg(parsed_features['image/encoded'])

    return parsed_features


def load_dataset(file_path):
    dataset=tf.data.TFRecordDataset(file_path)
    dataset=dataset.map(parse_function)
    return dataset

def go_positions_test():
    file_paths = [(r'C:\Users\Pathuru\Desktop\AI\AutoCNN-master\Dataset\train\go-pieces.tfrecord'), (r'C:\Users\Pathuru\Desktop\AI\AutoCNN-master\Dataset\test\go-pieces.tfrecord'), (r'C:\Users\Pathuru\Desktop\AI\AutoCNN-master\Dataset\valid\go-pieces.tfrecord')]  # Update paths accordingly
    datasets = [load_dataset(file) for file in file_paths]

    train_data = datasets[0]  # Assuming first tfrecord file is for training
    test_data = datasets[1]  # Assuming second tfrecord file is for testing
    valid_data = datasets[2]  # Assuming third tfrecord file is for validation
    
#    def inspect_dataset_shapes(dataset):
#        for images, labels in dataset.take(1):
#            print(f'Image batch dimensions: {images.shape}')
#            print(f'Label batch dimensions: {labels.shape}')
#            break  # Only look at the first batch

# Assuming `train_data` is the name of your dataset variable:
#        inspect_dataset_shapes(train_data)
#        inspect_dataset_shapes(test_data)
#        inspect_dataset_shapes(valid_data)

    
#    def inspect_dataset_elements(dataset):
#        for element in dataset.take(1):
#            print(element)

# Call the inspect function for your dataset
#        inspect_dataset_elements(train_data)
#        inspect_dataset_elements(test_data)
#        inspect_dataset_elements(valid_data)
    
    (train_data, valid_data), (test_data, test_data) = tf.keras.datasets.mnist.load_data()
    
    values = train_data.shape[0]
    
    data = {'x_train': train_data[:values], 'x_test': test_data, 'y_train': valid_data[:values], 'y_test': test_data}
    
    a=AutoCNN(5, 1, data)
    features_output=a.run()  # Assuming a.run() now returns feature vectors

    b=Prediction()
    policy_output, value_output=b.predict_move(features_output)
    
    print("Policy Output:", policy_output)
    print("Value Output:", value_output)

#def mnist_test():
#    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#    values = x_train.shape[0] // 2

#    data = {'x_train': x_train[:values], 'y_train': y_train[:values], 'x_test': x_test, 'y_test': y_test}

#    a = AutoCNN(5, 1, data)
#    a.run()


#def cifar10_test():
#    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

#    values = x_train.shape[0]

#    data = {'x_train': x_train[:values], 'y_train': y_train[:values], 'x_test': x_test, 'y_test': y_test}

#    a = AutoCNN(20, 10, data, epoch_number=10)
#    a.run()


if __name__ == '__main__':
    go_positions_test()
