import matplotlib.pyplot as plt
import brian2 as b2


def plot_rates(ERM: b2.PopulationRateMonitor, IRM: b2.PopulationRateMonitor) -> None:
    """

    :param ERM:
    :param IRM:
    :return:
    """
    plt.rcParams["figure.figsize"] = (20, 6)
    plt.plot(ERM.t / b2.ms, ERM.smooth_rate(window='flat', width=0.1 * b2.ms) * b2.Hz, color='r', label='Exc')
    plt.plot(IRM.t / b2.ms, IRM.smooth_rate(window='flat', width=0.1 * b2.ms) * b2.Hz, color='b', label='Inh')
    plt.ylabel('Rate')

    plt.legend()
    plt.show()

