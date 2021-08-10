import matplotlib
import matplotlib.pyplot as plt


def plot_positions(positions):
    plt.plot(positions[:, 0], positions[:, 1], 'o')
