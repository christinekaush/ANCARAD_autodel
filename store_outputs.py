# Builtin imports
from pprint import pprint
import sys
from pathlib import Path
import argparse

# External imports
import json
import yaml
#import sacred


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
        "model_version", help="Suffix number to use for experiment name.", type=str
    )
    parser.add_argument(
        "eval_metric",
        help="The evaluation metric to use when finding the best architecture.",
        type=str,
    )
    parser.add_argument(
        "--storefile",
        help="The name of the h5 file that the outputs are saved to",
        type=str,
    )
    parser.add_argument("--stepnum", help="The training step to use", type=int)
    parser.add_argument(
        "--skip_summary",
        help="If true, the performance summary is not computed",
        type=bool,
    )
    parser.add_argument("--dataset", help="test, train or val, specify which dataset to store outputs and generate summary from", default="val")

    args = parser.parse_args()
    return (
        Path(args.experiment),
        args.model_version,
        args.eval_metric,
        args.storefile,
        args.stepnum,
        args.skip_summary,
        args.dataset
    )


class SmartFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith("R|"):
            return text[2:].splitlines()
        return super()._split_lines(text, width)


if __name__ == "__main__":
    data_path, model_version, eval_metric, storefile, stepnum, skip_summary, dataset_type = (
        parse_arguments()
    )
    
    name = load_json(data_path / "experiment_params.json")["name"]

    dataset_params = load_json(data_path / "dataset_params.json")
    model_params = load_json(data_path / "model_params.json")
    trainer_params = load_json(data_path / "trainer_params.json")
    log_params = load_json(data_path / "log_params.json")
    experiment_params = load_json(data_path / "experiment_params.json")
    experiment_params["name"] += f"_{model_version}"
    print("Experiment_name:"+name)
    experiment_params["continue_old"] = True

    from scinets.utils.experiment import NetworkExperiment

    experiment = NetworkExperiment(
        experiment_params=experiment_params,
        model_params=model_params,
        dataset_params=dataset_params,
        trainer_params=trainer_params,
        log_params=log_params,
    )

    if stepnum is None:
        if not hasattr(experiment.evaluator, eval_metric):
            raise ValueError(
                "The final evaluation metric must be a "
                "parameter of the network evaluator."
            )


    best_it = stepnum
    if eval_metric is not None and stepnum is None:
        best_it, result, result_std = experiment.find_best_model("val", eval_metric)

    import pandas as pd
    evaluation_results = experiment.evaluate_model(dataset_type, best_it)
    summary = pd.DataFrame(columns=["result"], index=list(evaluation_results.keys()))

    dice_per_pat, dice_per_pat_std = experiment.get_dice_per_pat(dataset_type, dataset_params['arguments']['data_path'], best_it)

    print(f'{" All evaluation metrics at best iteration ":=^80s}')
    for metric, (result, result_std) in evaluation_results.items():
        summary["result"].loc[metric] = str(round(result, 3))+'+/-'+str(round(result_std,3))
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
    add["result"].loc["dice_per_pat"] = str(round(dice_per_pat, 3))+'+/-'+str(round(dice_per_pat_std,3))
    add["result"].loc['n_patients'] = n
    summary = pd.concat((summary, add), axis=0)
    summary.to_csv(f'res{name}{addon}_{best_it}.csv', sep=',', encoding='utf-8')
    print(80 * "=")

    if storefile is not None:
        print(f'{" Saving input and output to disk ":=^80s}')
        experiment.save_outputs(dataset_type, storefile, best_it)
        print("Outputs saved")