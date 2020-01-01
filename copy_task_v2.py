import tensorflow as tf
import numpy as np
import json
from model_v2 import CopyModel
from utils import generate_random_strings
import argparse


class SequenceCrossEntropyLoss(tf.keras.losses.Loss):
    eps = 1e-8

    def call(self, y_true, y_pred):
        return -tf.reduce_mean(   # cross entropy function
            y_true * tf.math.log(y_pred + self.eps) + (1 - y_true) * tf.math.log(1 - y_pred + self.eps)
        )


def train(config):
    model = CopyModel(
        batch_size=config['batch_size'],
        vector_dim=config['vector_dim'],
        model_type=config['model_type'],
        cell_params=config['cell_params'][config['model_type']]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    sequence_loss_func = SequenceCrossEntropyLoss()
    for batch_index in range(config['num_batches']):
        seq_length = tf.constant(np.random.randint(1, config['max_seq_length'] + 1), dtype=tf.int32)
        x = generate_random_strings(config['batch_size'], seq_length, config['vector_dim'])
        with tf.GradientTape() as tape:
            y_pred = model((x, seq_length))
            loss = sequence_loss_func(y_true=x, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
        if batch_index % 100 == 0:
            x = generate_random_strings(config['batch_size'], config['test_seq_length'], config['vector_dim'])
            y_pred = model((x, config['test_seq_length']))
            loss = sequence_loss_func(y_true=x, y_pred=y_pred)
            print("batch %d: loss %f" % (batch_index, loss))
            print("original string sample: ", x[0])
            print("copied string sample: ", y_pred[0])
        if batch_index % 5000 == 0:
            model.save_weights(config['save_dir'] + '_' + str(batch_index), save_format='tf')


def test(config, checkpoint_no):
    model = CopyModel(
        batch_size=config['batch_size'],
        vector_dim=config['vector_dim'],
        model_type=config['model_type'],
        cell_params=config['cell_params'][config['model_type']]
    )

    model.load_weights(config['save_dir'] + '_' + str(checkpoint_no))

    x = generate_random_strings(config['batch_size'], config['test_seq_length'], config['vector_dim'])
    y_pred = model((x, config['test_seq_length']))
    print("original string sample: ", x[0])
    print("copied string sample: ", y_pred[0])



def get_parser(config):
    parser = argparse.ArgumentParser(description='Train/test the ntm model')
    
    parser.add_argument('action_type', type=str, choices=['train', 'test'],
                        help='train/test')
    parser.add_argument('-c, --checkpoint_no', type=int, metavar='CN', default=995000,
                        help='Checkpoint to retrieve from. Only for testing',
                        dest='checkpoint')
    parser.add_argument('-l, --seq_length', type=int, metavar='L', 
                        default=config['test_seq_length'], dest='length',
                        help='Length of the testing data. Default is defined in the config file.')

    config['test_seq_length'] = parser.parse_args().length

    return parser.parse_args()



if __name__ == '__main__':
    with open('copy_task_config.json') as f:
        config = json.load(f)
    parser = get_parser(config)

    if parser.action_type == 'train':
        train(config)
    else:
        test(config, parser.checkpoint)
