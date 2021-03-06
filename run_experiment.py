
__author__ = "Yngve Mardal Moe"
__email__ = "yngve.m.moe@gmail.com"

"""
Modifications made by 
__author__ = "Christine Kiran Kaushal
__email__ = "christine.kiran@gmail.com
"""


# Builtin imports
from pprint import pprint
import sys
from pathlib import Path
import argparse

# External imports
import json
import yaml

import random


random.seed(999)

def _load_file_using_module(path, module):
    if isinstance(path, str):
        path = Path(path)
    with path.open() as f:
        return module.load(f)


def load_json(path):
    return _load_file_using_module(path, json)


def load_yaml(path):
    return _load_file_using_module(path, yaml)


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=SmartFormatter)
    parser.add_argument(
        "experiment",
        help="R|Path to folder with the JSON files specifying the parameters \n"
        "used in the experiment. The folder should contain the \n"
        "following files\n"
        "   - 'experiment_params.json'\n"
        "   - 'dataset_params.json'\n"
        "   - 'model_params.json'\n"
        "   - 'trainer_params.json'\n"
        "   - 'log_params.json'",
        type=str,
    )
    parser.add_argument(
        "num_steps", help="Number of training steps to perform.", type=int
    )
    parser.add_argument(
        "--eval",
        help="The evaluation metric to use when finding the best architecture.",
        type=str,
    )
    parser.add_argument(
        "--name", help="The name of the experiment used for logging.", type=str
    )

    args = parser.parse_args()
    return Path(args.experiment), args.num_steps, args.name, args.eval


class SmartFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith("R|"):
            return text[2:].splitlines()
        return super()._split_lines(text, width)


if __name__ == "__main__":
    import pandas as pd

    data_path, num_steps, name, eval_metric = parse_arguments()
    if name is None:
        name = load_json(data_path / "experiment_params.json")["name"]

    dataset_params = load_json(data_path / "dataset_params.json")
    model_params = load_json(data_path / "model_params.json")
    trainer_params = load_json(data_path / "trainer_params.json")
    log_params = load_json(data_path / "log_params.json")
    experiment_params = load_json(data_path / "experiment_params.json")
    experiment_params["name"] = name

    # Import here to prevent long startuptime if --help flag is used
    from scinets.utils.experiment import NetworkExperiment
    experiment = NetworkExperiment(
        experiment_params=experiment_params,
        model_params=model_params,
        dataset_params=dataset_params,
        trainer_params=trainer_params,
        log_params=log_params,
    )
    if eval_metric is not None:
        if not hasattr(experiment.evaluator, eval_metric):
            raise ValueError(
                "The final evaluation metric must be a parameter of the network evaluator."
            )
    next(experiment.dataset.train_data_reader.iterate_dataset_randomly())
    experiment.train(num_steps)

    if eval_metric is not None:
        best_it, result, result_std = experiment.find_best_model("val", eval_metric)
        #print(f'{" Final score ":=^80s}')
        #print(
        #    f" Achieved a {eval_metric:s} of {result:.3f}, with a standard "
        #    f"deviation of {result_std:.3f}"
        #)
        #print(f" This result was achieved at iteration {best_it}")
        #print(80 * "=")
        #final_result = result

        
    dataset_type = "val"
    evaluation_results = experiment.evaluate_model(dataset_type, best_it)
    summary = pd.DataFrame(columns=["result"], index=list(evaluation_results.keys()))
    
    dice_per_pat, dice_per_pat_std = experiment.get_dice_per_pat(dataset_type, dataset_params['arguments']['data_path'], best_it)

    print(f'{" All evaluation metrics at best iteration ":=^80s}')
    for metric, (result, result_std) in evaluation_results.items():
        summary["result"].loc[metric] = str(result)+'+/-'+str(result_std)
        print(
            f" Achieved a {metric:s} of {result:.3f}, with a standard "
            f"deviation of {result_std:.3f}"
        )

        addon = ""
        print(str(dataset_params['arguments']['data_path']))
        if '_MRI_' in str(dataset_params['arguments']['data_path']):
            n = 36
            #addon = "_35"
        else:
            n = 85
            #addon = "_85"
    
    add = pd.DataFrame(columns=["result"], index=["dice_per_pat", "n_patients"])
    add["result"].loc["dice_per_pat"] = str(round(dice_per_pat, 3))+'+/-'+str(dice_per_pat_std)
    add["result"].loc['n_patients'] = n
    summary = pd.concat((summary, add), axis=0)
    summary.to_csv(f'res{name}{addon}_{best_it}.csv', sep=',', encoding='utf-8')
    print(80 * "=")

    print(f'{" All evaluation metrics at best iteration ":=^80s}')
    for metric, (result, result_std) in evaluation_results.items():
        summary["result"].loc[metric] = str(result)+'+/-'+str(result_std)
        print(
            f" Achieved a {metric:s} of {result:.3f}, with a standard "
            f"deviation of {result_std:.3f}"
        )
