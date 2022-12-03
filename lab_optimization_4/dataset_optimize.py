import logging
import sys

import optuna
import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection

import matplotlib.pyplot as plt

from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_slice

def save_plot(*args, plot_func=None, filename=''):
    plot_func(*args)
    plt.savefig(filename, dpi=300)
    plt.clf()

def objective(trial):
    digs = sklearn.datasets.load_digits()
    classes = list(set(digs.target))
    train_x, valid_x, train_y, valid_y = sklearn.model_selection.train_test_split(
        digs.data, digs.target, test_size=0.25, random_state=0
    )

    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    clf = sklearn.linear_model.SGDClassifier(alpha=alpha)

    for step in range(100):
        clf.partial_fit(train_x, train_y, classes=classes)

        # Report intermediate objective value.
        intermediate_value = 1.0 - clf.score(valid_x, valid_y)
        trial.report(intermediate_value, step)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()

    return 1.0 - clf.score(valid_x, valid_y)


# Optimize problem
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=20)


# Show results
save_plot(study, plot_func=plot_optimization_history, filename='history.png')
save_plot(study, plot_func=plot_intermediate_values,  filename='inermediate_trials_values.png')
save_plot(study, plot_func=plot_parallel_coordinate,  filename='parallel_coordinate.png')
save_plot(study, plot_func=plot_slice,                filename='slice.png')
save_plot(study, plot_func=plot_param_importances,    filename='param_importances.png')
save_plot(study, plot_func=plot_edf,                  filename='edf.png')
