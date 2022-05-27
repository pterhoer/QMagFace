import matplotlib.pyplot as plt
import numpy as np
import tqdm
from datasets.paired import PairDataset
from experiments.plots.db_utils import get_db_colors, get_db_translation
from similarity.qmagface import QMagFace


def plot_training_vis(dataset_root, model, result_path):
    db_caps = get_db_translation()
    db_colors = get_db_colors()
    ylim_dict = {'magface100': (-0.05, 0.04), 'magface50': (-0.025, 0.025), 'magface18': (-0.06, 0.1)}
    ylim_dict_ticks = {'magface100': (-0.05, 0.04), 'magface50': (-0.05, 0.02), 'magface18': (-0.05, 0.1)}
    ylim_dict_steps = {'magface100': 0.01, 'magface50': 0.01, 'magface18': 0.05}
    xlim_dict = {'magface100': (0.2, 0.9), 'magface50': (0.15, 0.75), 'magface18': (0.3, 0.9)}
    plt.tight_layout()
    plot_range_x = np.linspace(*xlim_dict[model], num=2)
    dbs = ['lfw', 'colorferet', 'morph', 'adience']
    for db in tqdm.tqdm(dbs):
        pds = PairDataset((dataset_root, db))
        thresholds, omegas = QMagFace.get_thresholds_omegas(pds.embeddings, pds.pairs, pds.matches,
                                                            fmr_num=100, weights_num=100, max_ratio=2)
        plt.scatter(thresholds, omegas, color=db_colors[db], alpha=0.4, s=75)
        m, c = QMagFace.fit_line(thresholds, omegas)
        pred_w2s = m * plot_range_x + c
        plt.plot(plot_range_x, pred_w2s, label=db_caps[db], color=db_colors[db], lw=3)

    plt.subplots_adjust(left=0.2, bottom=0.15, top=0.95, right=0.925)
    plt.xlim(xlim_dict[model])
    plt.ylim(ylim_dict[model])
    plt.yticks(ticks=np.arange(*ylim_dict_ticks[model], step=ylim_dict_steps[model]))
    plt.ylabel(r'Optimal quality weight $\omega_{opt}$')
    plt.xlabel(r'Decision threshold $t$')
    plt.legend(loc='upper left')
    plt.savefig(result_path)
    plt.clf()
    # plt.show()


def main():
    root = '../../_data/single_images/'
    result_root = '../../_results/plots/'
    models = ['magface18', 'magface50', 'magface100']
    for model in models:
        plot_training_vis(root + model, model, result_root + model + '.pdf')


if __name__ == '__main__':
    main()
