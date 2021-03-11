#!/usr/bin/env python
import matplotlib.pyplot as plt


def my_plotter(dataset_1, dataset_2, xlabel, ylabel, fig_title, fig_size=(10,5),
               legend_label=None, x_lim=None, y_lim=None, save=False, save_path=None):

    def get_xlim(dataset_1):
        return 1.05*max(dataset_1)

    def get_ylim(dataset_2):
        return 1.05*max(dataset_2)

    if x_lim is None:
        x_lim = get_xlim(dataset_1)
    if y_lim is None:
        y_lim = get_ylim(dataset_2)

    fig, ax = plt.subplots(figsize=fig_size)
    plt.plot(dataset_1, dataset_2,label=legend_label)

    ax.tick_params(direction='in', length=6, width=2, top=True, right=True, which='major')
    ax.tick_params(direction='in', length=3, width=0.5, top=True, right=True, which='minor')

    ax.set_xlabel(xlabel,  size=16)
    ax.set_ylabel(ylabel, size=16)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    for tick in ax.xaxis.get_ticklabels():
        tick.set_fontsize(16)
        tick.set_fontname('serif')
    for tick in ax.yaxis.get_ticklabels():
        tick.set_fontsize(16)
        tick.set_fontname('serif')
    plt.xlim(0, x_lim)
    plt.ylim(0, y_lim)
    plt.legend(loc = "lower right", prop={'size': 12}, framealpha=1)
    plt.title(fig_title, size=18);
    if save is True:
        plt.savefig(save_path)
    plt.show()


def my_scatter(dataset_1, dataset_2, xlabel, ylabel, fig_title, fig_size=(10,5),
               legend_label=None, x_lim=None, y_lim=None, save=False, save_path=None):

    def get_xlim(dataset_1):
        return 1.05*max(dataset_1)

    def get_ylim(dataset_2):
        return 1.05*max(dataset_2)

    if x_lim is None:
        x_lim = get_xlim(dataset_1)
    if y_lim is None:
        y_lim = get_ylim(dataset_2)


    fig, ax = plt.subplots(figsize=fig_size)
    plt.scatter(dataset_1, dataset_2, label=legend_label, s=3)

    ax.tick_params(direction='in', length=6, width=2, top=True, right=True, which='major')
    ax.tick_params(direction='in', length=3, width=0.5, top=True, right=True, which='minor')

    ax.set_xlabel(xlabel,  size=14)
    ax.set_ylabel(ylabel, size=14)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    for tick in ax.xaxis.get_ticklabels():
        tick.set_fontsize(14)
        tick.set_fontname('serif')
    for tick in ax.yaxis.get_ticklabels():
        tick.set_fontsize(14)
        tick.set_fontname('serif')
    plt.legend(loc = "lower right", prop={'size': 12}, framealpha=1)
    plt.title(fig_title, size=16);
    plt.xlim(0, x_lim)
    plt.ylim(0, y_lim)
    if save is True:
        plt.savefig(save_path)
    plt.show();
