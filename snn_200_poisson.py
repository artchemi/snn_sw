import brian2 as b2
from brian2 import *
from keras.datasets import mnist
from tqdm import tqdm
from sklearn import metrics
import numpy as np
import pandas as pd
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from spikingjelly import visualizing
import utils
import networkx as nx
import os



# n_input = 28 * 28  # input layer
n_e = 100  # e - excitatory
n_i = n_e  # i - inhibitory

v_rest_e = -60. * mV  # v - membrane potential
v_reset_e = -65. * mV
v_thresh_e = -52. * mV

v_rest_i = -60. * mV
v_reset_i = -45. * mV
v_thresh_i = -40. * mV

taupre = 20 * ms
taupost = taupre
gmax = .05  # .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax

# Apre and Apost - presynaptic and postsynaptic traces, lr - learning rate
stdp = '''w : 1
    lr : 1 (shared)
    dApre/dt = -Apre / taupre : 1 (event-driven)
    dApost/dt = -Apost / taupost : 1 (event-driven)'''
pre = '''ge += w
    Apre += dApre
    w = clip(w + lr*Apost, 0, gmax)'''
post = '''Apost += dApost
    w = clip(w + lr*Apre, 0, gmax)'''


def generate_small_world(n_nodes, k, p):
    return nx.watts_strogatz_graph(n_nodes, k, p)


class Model:
    def __init__(self, n_input: int, n_hidden: int, dir_name: str, save_flag=False, debug=False) -> None:
        app = {}

        self.n_input = n_input
        self.n_hid = n_hidden

        self.dir_name = dir_name

        self.save_flag = save_flag

        # инициализация малого мира
        G = generate_small_world(self.n_hid, self.n_hid - 1, 0.5)

        # Входные изображения кодируются как скорость Пуассоновских генераторов
        app['PG'] = PoissonGroup(self.n_input, rates=np.zeros(self.n_input) * Hz, name='PG')

        # Группа возбуждающих нейронов
        neuron_e = '''
            dv/dt = (ge*(0*mV-v) + gi*(-100*mV-v) + (v_rest_e-v)) / (100*ms) : volt
            dge/dt = -ge / (5*ms) : 1
            dgi/dt = -gi / (10*ms) : 1
            '''
        app['EG'] = NeuronGroup(self.n_hid, neuron_e, threshold='v>v_thresh_e', refractory=5 * ms, reset='v=v_reset_e',
                                method='euler', name='EG')
        app['EG'].v = v_rest_e - 20. * mV

        if (debug):
            app['ESP'] = SpikeMonitor(app['EG'], name='ESP')
            app['ESM'] = StateMonitor(app['EG'], ['v'], record=True, name='ESM')
            app['ERM'] = PopulationRateMonitor(app['EG'], name='ERM')

        # Группа ингибирующих нейронов
        neuron_i = '''
            dv/dt = (ge*(0*mV-v) + (v_rest_i-v)) / (10*ms) : volt
            dge/dt = -ge / (5*ms) : 1
            '''
        app['IG'] = NeuronGroup(self.n_hid, neuron_i, threshold='v>v_thresh_i', refractory=2 * ms, reset='v=v_reset_i',
                                method='euler', name='IG')
        app['IG'].v = v_rest_i - 20. * mV

        if (debug):
            app['ISP'] = SpikeMonitor(app['IG'], name='ISP')
            app['ISM'] = StateMonitor(app['IG'], ['v'], record=True, name='ISM')
            app['IRM'] = PopulationRateMonitor(app['IG'], name='IRM')

        # poisson generators one-to-all excitatory neurons with plastic connections
        app['S1'] = Synapses(app['PG'], app['EG'], stdp, on_pre=pre, on_post=post, method='euler', name='S1')
        app['S1'].connect()
        app['S1'].w = 'rand()*gmax'  # random weights initialisation
        app['S1'].lr = 1  # enable stdp

        # здесь надо добавить другое правило обучения, а не STDP
        app['S_small_world'] = Synapses(app['EG'], app['EG'], stdp, on_pre=pre, on_post=post, method='euler', name='S_small_world')

        for (u, v) in G.edges():
            app['S_small_world'].connect(j=u, i=v)
            app['S_small_world'].w = 'rand()*gmax'  # здесь нужно сделать так, чтобы генерировались рандомные веса на порядок меньше, чем основные

        app['S_small_world'].lr = 1  # enable stdp

            # excitation neurons one-to-one inhibitory neurons
        app['S2'] = Synapses(app['EG'], app['IG'], 'w : 1', on_pre='ge += w', name='S2')
        app['S2'].connect(j='i')
        app['S2'].delay = 'rand()*10*ms'
        app['S2'].w = 3  # very strong fixed weights to ensure corresponding inhibitory neuron will always fire

        # inhibitory neurons one-to-all-except-one excitatory neurons
        app['S3'] = Synapses(app['IG'], app['EG'], 'w : 1', on_pre='gi += w', name='S3')
        app['S3'].connect(condition='i!=j')
        app['S3'].delay = 'rand()*5*ms'
        app['S3'].w = .03  # weights are selected in such a way as to maintain a balance between excitation and ibhibition

        if (debug):
            debug_params = {}
            # app['S_SW_M'] = StateMonitor(app['S_small_world'], ['w', 'Apre', 'Apost'], record=app['S_small_world'], name='S_SW_M')
            # app['S1M'] = StateMonitor(app['S1'], ['w', 'Apre', 'Apost'], record=app['S1'][0, :], name='S1M')

        self.net = Network(app.values())
        self.net.run(0 * second)

    def __getitem__(self, key):
        return self.net[key]

    def train(self, X, epoch=1):
        self.net['S1'].lr = 1  # stdp on

        for ep in tqdm(range(epoch)):

            if self.save_flag == True:
                self.net.store('train', f'{self.dir_name}/chk_{ep}.b2')

            # сохранение межслойных весов
            with open(f'{self.dir_name}/weights/weight_data{ep}.npy', 'wb') as file:
                np.save(file, np.array(self.net['S1'].w).reshape((784, self.n_hid)))

            for idx in tqdm(range(len(X))):
                # active mode
                self.net['PG'].rates = X[idx].ravel() * Hz
                self.net.run(0.35 * second)

                # passive mode
                self.net['PG'].rates = np.zeros(self.n_input) * Hz
                self.net.run(0.15 * second)

    def evaluate(self, X, chk: str):
        self.net['S1'].lr = 0  # stdp off

        if chk != None:
            self.net.restore(name='train', filename=chk)

        features = []
        for idx in tqdm(range(len(X))):
            # rate monitor to count spikes
            mon = SpikeMonitor(self.net['EG'], name='RM')
            self.net.add(mon)

            # active mode
            self.net['PG'].rates = X[idx].ravel() * Hz
            self.net.run(0.35 * second)

            # spikes per neuron foreach image
            features.append(np.array(mon.count, dtype=int8))

            # passive mode
            self.net['PG'].rates = np.zeros(self.n_input) * Hz
            self.net.run(0.15 * second)

            self.net.remove(self.net['RM'])

        features = np.array(features)
        # features = features.reshape((10, 10))

        return features


def plot_w(S1M):
    plt.rcParams["figure.figsize"] = (20, 10)
    subplot(311)
    plot(S1M.t / ms, S1M.w.T / gmax)
    ylabel('w / wmax')
    subplot(312)
    plot(S1M.t / ms, S1M.Apre.T)
    ylabel('apre')
    subplot(313)
    plot(S1M.t / ms, S1M.Apost.T)
    ylabel('apost')
    tight_layout()
    show();


def plot_v(ESM, ISM, neuron=13):
    plt.rcParams["figure.figsize"] = (20, 6)
    cnt = -50000  # tail
    plot(ESM.t[cnt:] / ms, ESM.v[neuron][cnt:] / mV, label='exc', color='r')
    plot(ISM.t[cnt:] / ms, ISM.v[neuron][cnt:] / mV, label='inh', color='b')
    plt.axhline(y=v_thresh_e / mV, color='pink', label='v_thresh_e')
    plt.axhline(y=v_thresh_i / mV, color='silver', label='v_thresh_i')
    legend()
    ylabel('v')
    show();


def plot_rates(ERM, IRM):
    plt.rcParams["figure.figsize"] = (20, 6)
    plot(ERM.t / ms, ERM.smooth_rate(window='flat', width=0.1 * ms) * Hz, color='r')
    plot(IRM.t / ms, IRM.smooth_rate(window='flat', width=0.1 * ms) * Hz, color='b')
    ylabel('Rate')
    show();


def plot_spikes(ESP, ISP):
    plt.rcParams["figure.figsize"] = (20, 6)
    plot(ESP.t / ms, ESP.i, '.r')
    plot(ISP.t / ms, ISP.i, '.b')
    ylabel('Neuron index')
    show();


def test0(train_items=30):
    '''
    STDP visualisation
    '''
    seed(0)

    model = Model(debug=True)
    print(X_train[:train_items])
    model.train(X_train[:train_items], epoch=1)

    plot_w(model['S1M'])
    plot_v(model['ESM'], model['ISM'])
    plot_rates(model['ERM'], model['IRM'])
    plot_spikes(model['ESP'], model['ISP'])


def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # simplified classification (0 1 and 8)
    # X_train = X_train[(y_train == 1) | (y_train == 0) | (y_train == 8)]
    # y_train = y_train[(y_train == 1) | (y_train == 0) | (y_train == 8)]
    # X_test = X_test[(y_test == 1) | (y_test == 0) | (y_test == 8)]
    # y_test = y_test[(y_test == 1) | (y_test == 0) | (y_test == 8)]

    # pixel intensity to Hz (255 becoms ~63Hz)
    X_train = X_train / 4
    X_test = X_test / 4

    X_test = X_test[:1]
    y_test = y_test[:1]

    model = Model(True)
    assign_items = len(X_test)  # 60k

    seed(0)

    f_test = model.evaluate(X_test)

    clf = RandomForestClassifier(max_depth=4, random_state=0)
    clf.fit(f_test, y_test)

    print(clf.score(f_test, y_test))

    y_pred = clf.predict(f_test)
    conf_m = confusion_matrix(y_pred, y_test)
    print(conf_m)

    model.net.store('train', 'train.b2')

    # --- Датасет с данными о модели ---
    # time - время
    # exc_rate, inh_rate - частота спайкования возбуждающих и ингибирующих нейронов

    data_dict = {'time': model['ERM'].t / b2.ms,
                 'exc_rate': model['ERM'].smooth_rate(window='flat', width=0.1 * b2.ms) / b2.Hz,
                 'inh_rate': model['IRM'].smooth_rate(window='flat', width=0.1 * b2.ms) / b2.Hz}

    # (n_exc = n_inh)
    # n_exc_{i}, n_inh_{i} - мембранный потенциал возбуждающих и ингибирующих нейронов

    for i in range(n_e):
        data_dict[f'n_exc_{i}'] = model['ESM'].v[i] / mV
        data_dict[f'n_inh_{i}'] = model['ISM'].v[i] / mV

    labels = []
    for label in y_test:
        labels.append([label] * 5000)
    data_dict['labels'] = list(itertools.chain(*labels))

    dataframe = pd.DataFrame(data=data_dict)
    dataframe.to_csv('data.csv', index=False)

    print(model['ESM'].v / b2.mV)

    visualizing.plot_2d_heatmap(array=np.asarray(model['ESM'].v / b2.mV).T, title='Membrane Potentials', xlabel='Simulating Step',
                                ylabel='Neuron Index', int_x_ticks=True, x_max=5000, dpi=200)

    # visualizing.plot_2d_bar_in_3d(np.asarray(model['ESM'].v[0:6] / b2.mV).T, title='voltage of neurons', xlabel='neuron index',
    #                               ylabel='simulating step', zlabel='voltage', int_x_ticks=True, int_y_ticks=True,
    #                               int_z_ticks=True, dpi=200)

    plt.show()

    # utils.plot_rates(model['ERM'], model['IRM'])

    # model.net.restore()


if __name__ == '__main__':
    main()
