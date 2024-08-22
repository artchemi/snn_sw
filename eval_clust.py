from snn_200_poisson import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import argparse
import time


parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')


parser.add_argument('--checkpoint', default=None, type=int, 
                    help='С какой эпохи брать чекпоинт')

args = parser.parse_args()


def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train[(y_train == 0) | (y_train == 1) | (y_train == 2) | (y_train == 3)]
    X_train = X_train[:1]

    current_time = time.time()
    local_time = time.localtime(current_time)
    formatted_time = time.strftime('%d_%m_%Y_%H%M%S', local_time)

    dir_name = f'data_{formatted_time}'

    print(f'Train len: {len(X_train)}')

    model = Model(784, 200, dir_name, False, True)

    features = model.evaluate([X_train], args.checkpoint)
    features = features.reshape((10, 10))

    plt.imshow(features)
    plt.show()




if __name__ == '__main__':
    main()