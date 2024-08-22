import numpy as np

#Функция, сопоставляющая лейблы классов и нейроны
#
#Аргументы:
#x, y -- датасет, на котором будем проводить сопоставление лейблов и нейронов
#    размера не менее ex_num для каждого класса, в x -- примеры, в y -- лейблы
#    индексация должна совпадать
#labels -- список возможныых лейблов (для mnist, например -- кортеж от 0 до 9)
#rate_func -- функция, возвращающая рейты нейронов
#    фактически, predict
#    должна уметь работать при неуказанных параметрах, кроме датасета
#neuron_num -- количество нейронов, рейты которых записываются
#ex_num -- количество примеров для одного класса
#    чем больше, тем лучше карта на выходе должна получиться, но сильно увеличивает время счёта
#
#Возвращает:
#neurons_rates -- список массивов "весов" нейрона при решениии об отнесении набора рейтов со всех нейронов к классу
#    первый индекс: выбор пары массив-лейбл
#    второй индекс: массив весов при [0], лейбл при [1]
#all_classes_rates -- рейты всех нейронов, всех классов, список двумерных массивов
#    первый индекс: выбор класса
#    второй индекс: выбор примера
#    можно использовать для визуализации частот спайков с помощью heatmap
def assign_labels(x, y, labels, rate_func, neuron_num, ex_num = 10):
    all_classes_rates = []
    all_classes_neurons = []
    for class_num in range(len(labels)):
        class_rates = []
        class_neurons = []
        for example_num in range(ex_num):
            rates = rate_func((x[(y == labels[class_num])][example_num],))
            neurons = np.where(rates == np.max(rates))[1]
            class_rates.append(rates)
            class_neurons.append(neurons)
        all_classes_rates.append(np.array(class_rates))
        all_classes_neurons.append(class_neurons)

    neurons_rates = []
    
    for class_num in range(len(labels)):
        neurons_rate = np.zeros(neuron_num)
        for example_neurons in all_classes_neurons[class_num]:
            for n in range(neurons_rate.size):
                if n in example_neurons:
                    neurons_rate[n] += 1/len(all_classes_neurons[class_num])
        neurons_rate = neurons_rate/np.max(neurons_rate)
        neurons_rate = neurons_rate/np.sum(neurons_rate)
        neurons_rates.append((neurons_rate, labels[class_num]))
        
    return neurons_rates, all_classes_rates

#Функция, относящая распределение рейтов множества нейронов к одному классу
#
#Аргументы:
#rates -- рейты нейронов, одномерный массив
#neurons_rating -- список массивов "весов" нейрона
#    первый индекс: выбор пары массив-лейбл
#    второй индекс: массив весов при [0], лейбл при [1]
#
#Возвращает:
#max_sum_label -- лейбл класса для данного набора рейтов
def guess_label(rates, neurons_rating):
    max_sum = 0
    max_sum_label = neurons_rating[0][0]
    for r in neurons_rating:
        recall = np.sum(rates * r[0])
        if recall > max_sum:
            max_sum = recall
            max_sum_label = r[1]

    return max_sum_label

#Функция, относящая распределение рейтов множества нейронов к классам
#
#Аргументы:
#rates -- рейты нейронов, для множества примеров, двумерный массив
#    первый индекс: рейты нейронов для конкретного примера
#    второй индекс: индекс нейрона
#neurons_rating -- список массивов "весов" нейрона
#    первый индекс: выбор пары массив-лейбл
#    второй индекс: массив весов при [0], лейбл при [1]
#
#Возвращает:
#guessed_label -- лейблы классов для данного набора рейтов

def guess_labels(rates, neuron_rating):
    guessed_labels = []
    for rate in rates:
        guessed_labels.append(guess_label(rate, neuron_rating))
    guessed_labels = np.array(guessed_labels)

    return guessed_labels
