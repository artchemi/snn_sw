from snn_200_poisson import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import argparse
import os
import time

import brian2 as b2


parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')


parser.add_argument('-e', '--epochs', default=10, type=int, 
                    help='Количество эпох для обучения')

parser.add_argument('--hidden', default=100, type=int, 
                    help='Количество нейронов в скрытом слое')

args = parser.parse_args()


def main():
    parser.print_help()

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train[(y_train == 0) | (y_train == 1) | (y_train == 2) | (y_train == 3)]
    X_train = X_train[:1]

    current_time = time.time()
    local_time = time.localtime(current_time)
    formatted_time = time.strftime('%d_%m_%Y_%H%M%S', local_time)

    dir_name = f'data_{formatted_time}'

    print(f'Train len: {len(X_train)}')
    
    os.mkdir(dir_name)
    os.mkdir(dir_name + '/weights')

    model = Model(784, args.hidden, dir_name, False, True)

    model.train(X_train, args.epochs)

    print(model['ERM'].smooth_rate(window='flat', width=0.1 * b2.ms) / b2.Hz)
    

if __name__ == '__main__':
    main()