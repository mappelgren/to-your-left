import argparse
import glob
import json
import os
import pprint
import re
from dataclasses import dataclass


@dataclass
class Experiment:
    folder_name: str
    variable_string: str
    dataset: str
    variables: dict
    runs: list[dict]
    last_run: dict


def sort_by_variable(runs):
    return dict(
        sorted(
            runs.items(),
            key=lambda x: x[1][1],
            reverse=False,
        )
    )


sorts = {"metric": sort_by_metric, "variable": sort_by_variable}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folder",
        type=str,
        help="path to the results folders (can contain wildcards)",
    )

    parser.add_argument(
        "--variables",
        nargs="+",
        help="variables in the same order as in the folder name",
    )

    parser.add_argument(
        "--variable_order",
        nargs="+",
        type=int,
        help="variables in the same order as in the folder name",
    )

    parser.add_argument(
        "--sort",
        choices=sorts.keys(),
        help="sorting of the results",
    )

    args = parser.parse_args()
    if len(args.variable_order) == 0:
        args.variable_order = range(len(args.variables))
    if len(args.variable_order) != len(args.variables):
        raise ValueError(
            f"Number of variables and orders don't match: {args.variable_order}, {args.variables}"
        )
    print(args)

    experiments: list[Experiment] = []
    all_results = {}
    for folder in glob.glob(args.folder):
        folder_name = folder.split("/")[-1]

        pattern = r"_([^_]+)((_\d+)*)$"
        match = re.search(pattern, folder_name)
        if match:
            dataset = match.group(1)
            variables = [int(var) for var in match.group(2).split("_") if var != ""]
            if len(variables) != len(args.variables):
                raise ValueError(f"Number of Variables don't match: {folder_name}")
            ordered_variables = [variables[o] for o in args.variable_order]

        variable_string = "_".join(map(str, ordered_variables))

        with open(os.path.join(folder, "log.txt"), "r", encoding="utf-8") as f:
            runs = []
            for line in f:
                if line.startswith("{"):
                    epoch = json.loads(line)
                    if epoch.get("mode", "test") == "train":
                        continue

                    flattened = {}
                    for k, v in epoch.items():
                        if isinstance(v, dict):
                            for sub_k, sub_v in v.items():
                                flattened[f"{k}_{sub_k}"] = sub_v
                        else:
                            flattened[k] = v

                    runs.append(flattened)

            if len(runs) != 0:
                experiments.append(
                    Experiment(
                        folder_name=folder_name,
                        variable_string=variable_string,
                        dataset=dataset,
                        variables=variables,
                        runs=runs,
                        last_run=runs[-1],
                    )
                )

    all_metrics = set(experiment.last_run.keys() for experiment in experiments)
    sorted_metrics = list(sorted(all_metrics))
    print(list(enumerate(sorted_metrics)))

    choice = input("choose important metrics (separate by comma):")
    chosen_metrics = [sorted_metrics[int(c)] for c in choice.split(",")]

    sorted_experiments = list(
        sorted(
            experiments,
            key=lambda experiment: experiment.variables,
            reverse=False,
        )
    )

    all_datasets = set(experiment.dataset for experiment in experiments)
    all_variables = set(experiment.variables for experiment in experiments)
    all_variations = set(experiment.variable_string for experiment in experiments)

    number_datasets = len(all_datasets)
    number_variables = len(all_variables)
    number_metrics = len(chosen_metrics)

    dataset_header = " & ".join(
        [
            rf"\multicolumn{{{number_metrics}}}{{c}}{{\textbf{{{dataset}}}}}"
            for dataset in all_datasets
        ]
    )
    cmidrule_header = "".join(
        [
            rf"\cmidrule(lr){{{number_variables + index * number_metrics +1}-{(number_variables + index * number_metrics) + number_metrics}}}"
            for index in range(number_datasets)
        ]
    )

    metric_header_per_dataset = " & ".join(
        [rf"\textbf{{{metric}}}" for metric in chosen_metrics]
    )
    metric_header = f" & {metric_header_per_dataset}" * number_datasets

    result_lines = []
    for run_name, _, variables in runs:
        variable_string = " & ".join([f"{{{var}}}" for var in variables])
        pprint.pprint(sorted_runs.values())
        result_string = " & ".join(
            [
                f"{{{dataset[run_name][0][metric]}}}"
                for dataset in sorted_runs.values()
                for metric in chosen_metrics
            ]
        )
        result_lines.append(rf"{variable_string} & {result_string} \\")

    results_block = "\n".join(result_lines)

    latex_stub = rf"""
\begin{{table}}[ht]
    \centering
    \begin{{tabular}}{{{'c'*number_variables}{f"|{'c'*number_metrics}"*number_datasets}}}
        \toprule
        {' &' *number_variables}{dataset_header} \\  {cmidrule_header}
        {' & '.join([args.variables[o].replace('_', '-') for o in args.variable_order])}     {metric_header}    \\\midrule
        {results_block}
        \bottomrule
    \end{{tabular}}
    \caption{{test}}
    \label{{test}}
\end{{table}}
"""

    print(latex_stub)
