import numpy as np
import random
import matplotlib.pyplot as plt

#Для создания картинки с паттерном, при создании объекта указать ориентацию
#может быть 'vertical' или 'horizontal'
#также можно там же указать уровень шума
#шум равномерный
#Цвет пикселя на картике закодирован числом от 0 до 1
#Для доступа к сгенерированному изображению: имя_объекта.M

class Picture:
    
    def __first_dot__(self):
        self.root_indexes = random.randint(0, 2), random.randint(0, 2)

        self.M[self.root_indexes[0], self.root_indexes[1]] = 1

    def __neighbour_dot__(self, prev_dig):
        if random.randint(0, 1) == 1:
            new_dig = prev_dig + 1
        else:
            new_dig = prev_dig - 1
        if new_dig > 2:
            new_dig -= 2
        elif new_dig < 0:
            new_dig += 2
        return new_dig

    def __second_dot__(self):
        second_dot_indexes = self.__neighbour_dot__(self.root_indexes[0]), self.root_indexes[1] 

        self.M[second_dot_indexes[0], second_dot_indexes[1]] = 1

    def __third_dot__(self):
        if random.randint(0, 1) == 1:
            index = np.where(self.M[:, self.root_indexes[1]] == 0)[0]
            self.M[index, self.root_indexes[1]] = 1

    def __second_line__(self):
        possible_indexes = [0, 1, 2]
        possible_indexes.remove(self.root_indexes[1])
        if random.randint(0, 1) == 1:
            new_index = random.choice(possible_indexes)
            if random.randint(0, 1) == 1:
                self.M[:, new_index] = self.M[:, self.root_indexes[1]]
            else:
                self.M[:, new_index] = np.flip(self.M[:, self.root_indexes[1]])

    def __add_noise__(self, level):
        for i in range(3):
            for j in range(3):
                if self.M[i, j] == 0:
                    self.M[i, j] += random.uniform(0, level)
                else:
                    self.M[i, j] -= random.uniform(0, level)

    def __check_square__(self):
        if np.sum(self.M) >= 4:
            for i in range(2):
                if (np.sum(self.M[:, i]) == 2) and (np.sum(self.M[:, i + 1]) == 2):
                    return True
        return False
            
    def __init__(self, orientation, noise_level = 0.35):
        if (orientation != 'horizontal') and (orientation != 'vertical'):
            print('Orientation must be horizontal or vertical! Not', orientation, '!')
        else:
            self.orientation = orientation
            self.M = np.zeros((3,3))

        square = True
        while(square):
            self.__first_dot__()
            self.__second_dot__()
            self.__third_dot__()
            self.__second_line__()
            square = self.__check_square__()

        self.__add_noise__(noise_level)

        if self.orientation == 'horizontal':
            self.M = self.M.T


def main():
    pic = Picture('vertical')
    print(pic.M)

    plt.imshow(pic.M)
    plt.show()

    


if __name__ == '__main__':
    main()