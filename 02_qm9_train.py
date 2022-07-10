#!/usr/bin/env python3

import argparse
from argparse import HelpFormatter, ArgumentDefaultsHelpFormatter
from datetime import datetime
import json
from operator import attrgetter
from pathlib import Path
import pickle
from pprint import pprint
import sys

sys.path.append(str(Path.cwd().parent))

from xas_nne.ml import Ensemble  # noqa

_now = datetime.now()
NOW = _now.strftime("%y%m%d")
NOW_DATETIME = _now.strftime("%y%m%d-%H%M%S")


def save_json(d, path):
    with open(path, 'w') as outfile:
        json.dump(d, outfile, indent=4, sort_keys=True)


def read_json(path):
    with open(path, 'r') as infile:
        dat = json.load(infile)
    return dat


def print_data_diagnostics(data):
    keys = ["train", "val", "test"]
    print("Data information as loaded")
    for key in keys:
        x = data[key]['x'].shape
        y = data[key]['y'].shape
        print(f"{key} : x~{x} | y~{y}")


# https://stackoverflow.com/questions/
# 12268602/sort-argparse-help-alphabetically
class SortingHelpFormatter(ArgumentDefaultsHelpFormatter, HelpFormatter):
    def add_arguments(self, actions):
        actions = sorted(actions, key=attrgetter('option_strings'))
        super(SortingHelpFormatter, self).add_arguments(actions)
        # self._max_help_position = 40
        # self._width = 100


def parser(sys_argv):
    ap = argparse.ArgumentParser(formatter_class=SortingHelpFormatter)
    ap.add_argument(
        '--data-path', dest='data_path', required=True,
        help='The path to the pickled data'
    )
    ap.add_argument(
        '--ensemble-name', dest='ensemble_name', default=None,
        help='The name of the ensemble.'
    )
    ap.add_argument(
        '-n', '--n-estimators', dest='n_estimators', type=int, default=30,
        help='The number of estimators to use during training.'
    )
    ap.add_argument(
        '--random-kwargs-json', dest='random_kwargs_json', type=str,
        default="random_kwargs.json",
        help='Parameter file for initializing an ensemble from random '
        'architectures.'
    )
    ap.add_argument(
        '--architecture-seed', dest='architecture_seed', type=int, default=123,
        help='Seeds the random architecture generation process.'
    )
    ap.add_argument(
        '--ensemble-index', dest='ensemble_index', type=int, default=0,
        help='The ensemble index.'
    )
    ap.add_argument(
        '--max-epochs', dest='max_epochs', type=int, default=5000
    )
    ap.add_argument(
        '--max-jobs', dest='max_jobs', type=int, default=3,
        help='Maximum number of simultaneous trainings to execute. Only works '
        'well for parallel training on a *single* GPU.'
    )
    ap.add_argument(
        '--downsample-prop', dest='downsample_prop', type=float, default=0.9,
        help='The percent of the training data to use during training. Used '
        'for bootstrapping during ensemble training.'
    )
    ap.add_argument(
        '--print-every-epoch', dest='print_every_epoch', type=int, default=100
    )
    ap.add_argument(
        '--dryrun', dest='dryrun', default=False,
        action="store_true"
    )

    return ap.parse_args(sys_argv)


if __name__ == '__main__':
    args = parser(sys.argv[1:])
    args_as_dict = vars(args)
    print(f"Now is {NOW_DATETIME}")
    pprint(args_as_dict)

    if args.ensemble_name is None:
        tmp = str(Path(args.data_path).stem)
        args.ensemble_name \
            = str(Path("Ensembles") / Path(f"{NOW}-{tmp}-ensemble"))
    else:
        args.ensemble_name = str(f"{NOW}-{args.ensemble_name}")
    print(f"Ensemble path set to {args.ensemble_name}")

    from_random_architecture_kwargs = read_json(args.random_kwargs_json)
    print("Random kwargs architecture parameters")
    pprint(from_random_architecture_kwargs)

    # Setup the ensemble
    ensemble = Ensemble.from_random_architectures(
        root=args.ensemble_name,
        n_estimators=args.n_estimators,
        seed=args.architecture_seed,
        from_random_architecture_kwargs=from_random_architecture_kwargs,
    )

    # Load the data and print diagnostic information
    data = pickle.load(open(args.data_path, "rb"))
    print_data_diagnostics(data)

    if args.dryrun:
        exit(0)

    # Execute training
    ensemble.train_ensemble_parallel(
        training_data=data["train"],
        validation_data=data["val"],
        ensemble_index=args.ensemble_index,
        epochs=args.max_epochs,
        n_jobs=args.max_jobs,
        downsample_training_proportion=args.downsample_prop,   # Bootstrapping
        print_every_epoch=args.print_every_epoch
    )

    # Save the ensemble metadata
    d = ensemble.as_dict()
    path = Path(args.ensemble_name) / Path("Ensemble.json")
    save_json(d, path)

    # Save the other metadata
    path = Path(args.ensemble_name) / Path("args.json")
    save_json(args_as_dict, path)
    path = Path(args.ensemble_name) / Path("random_architecture_kwargs.json")
    save_json(from_random_architecture_kwargs, path)
