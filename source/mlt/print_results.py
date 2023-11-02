import argparse
import glob
import json
import os
import re
from dataclasses import dataclass


@dataclass
class Experiment:
    folder_name: str
    dataset: str
    variables: tuple
    runs: list[dict]

    @property
    def last_run(self):
        return self.runs[-1]


def calculate_f1(run):
    if "word_by_word_precision" in run.keys() and "word_by_word_recall" in run.keys():
        run["word_by_word_f1"] = (
            2
            * (run["word_by_word_precision"] * run["word_by_word_recall"])
            / (run["word_by_word_precision"] + run["word_by_word_recall"])
        )

    for attribute in ["shape", "size", "color"]:
        if (
            f"precision_{attribute}" in run.keys()
            and f"recall_{attribute}" in run.keys()
        ):
            run[f"f1_{attribute}"] = (
                (
                    2
                    * (run[f"precision_{attribute}"] * run[f"recall_{attribute}"])
                    / (run[f"precision_{attribute}"] + run[f"recall_{attribute}"])
                )
                if run[f"precision_{attribute}"] + run[f"recall_{attribute}"] != 0
                else float("nan")
            )

    return run


def calculate_metrics_by_attribute(run):
    attributes = {
        "shape": ["sphere", "cube", "cylinder"],
        "size": [
            "large",
            "small",
        ],
        "color": [
            "gray",
            "red",
            "blue",
            "green",
            "brown",
            "purple",
            "cyan",
            "yellow",
        ],
    }

    for metric in ["precision", "recall", "accuracy"]:
        for attribute, values in attributes.items():
            for value in values:
                if f"{metric}_by_word_{value}" not in run.keys():
                    return run

            run[f"{metric}_{attribute}"] = sum(
                run[f"{metric}_by_word_{value}"] for value in values
            ) / len(values)

    return run


def calculate_metrics(run):
    run = calculate_metrics_by_attribute(run)
    run = calculate_f1(run)
    return run


def to_percent(number):
    rounded = str(round(number * 100, 2)).replace(".", ",").rstrip("0").rstrip(",")
    return rf"{rounded}\%"


def to_number(number):
    return rf"{str(round(number, 2)).rstrip('0.').replace('.', ',')}"


metric_types = {
    "%": to_percent,
    "num": to_number,
}


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
        help="order of the chosen variables",
    )
    parser.add_argument(
        "--metric_types",
        nargs="+",
        choices=metric_types.keys(),
        help="type of the variable in order of --variables",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        help="datasets in order in which they should appear in the table",
    )

    args = parser.parse_args()
    if not args.variable_order:
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

            if dataset not in args.datasets:
                continue

            variables = [int(var) for var in match.group(2).split("_") if var != ""]
            if len(variables) != len(args.variables):
                raise ValueError(f"Number of Variables don't match: {folder_name}")
            ordered_variables = tuple(variables[o] for o in args.variable_order)

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
                                flattened[f"{k}_{sub_k}"] = float(sub_v)
                        else:
                            flattened[k] = float(v)

                    extended = calculate_metrics(flattened)

                    runs.append(extended)

            if len(runs) != 0:
                experiments.append(
                    Experiment(
                        folder_name=folder_name,
                        dataset=dataset,
                        variables=ordered_variables,
                        runs=runs,
                    )
                )

    for index, folder in enumerate(
        [experiment.folder_name for experiment in experiments]
    ):
        print(f"{index}: {folder}")
    excluded_input = input("exclude folders (separated by comma, ranges with hyphen):")
    if len(excluded_input) != 0:
        excluded_indices = []
        for index in excluded_input.split(","):
            if "-" in index:
                range_indices = index.split("-")
                excluded_indices.extend(
                    range(int(range_indices[0]), int(range_indices[1]) + 1)
                )
            else:
                excluded_indices.append(int(index))

        experiments = [
            experiment
            for index, experiment in enumerate(experiments)
            if index not in excluded_indices
        ]

    all_metrics = set(
        metric for experiment in experiments for metric in experiment.last_run.keys()
    )
    sorted_metrics = list(sorted(all_metrics))
    for index, metric in enumerate(sorted_metrics):
        print(f"{index}: {metric}")
    choice = input("choose important metrics (separated by comma):").split(",")

    if not args.metric_types:
        args.metric_types = ["%"] * len(choice)
    chosen_metrics = {
        sorted_metrics[int(c)]: metric_types[metric_type]
        for c, metric_type in zip(choice, args.metric_types)
    }

    all_datasets = args.datasets
    all_variables = set(args.variables[i] for i in args.variable_order)
    sorted_variations = sorted(set(experiment.variables for experiment in experiments))

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
        [rf"\textbf{{{metric.replace('_', '-')}}}" for metric in chosen_metrics]
    )
    metric_header = f" & {metric_header_per_dataset}" * number_datasets

    result_lines = []
    for variation in sorted_variations:
        variation_experiments = [
            experiment
            for experiment in experiments
            if experiment.variables == variation
        ]

        variable_string = "        " + " & ".join(
            [f"{{{variable}}}" for variable in variation]
        )

        result_string = " & ".join(
            [
                rf"{{{metric_type(experiment.last_run[metric])}}}"
                for dataset in all_datasets
                for experiment in variation_experiments
                for metric, metric_type in chosen_metrics.items()
                if experiment.dataset == dataset
            ]
        )
        result_lines.append(rf"{variable_string} & {result_string} \\")

    results_block = "\n".join(result_lines)

    latex_stub = rf"""\begin{{table}}[ht]
    \centering
    \begin{{tabular}}{{{'c'*number_variables}{f"|{'c'*number_metrics}"*number_datasets}}}
        \toprule
        {' &' *number_variables}{dataset_header} \\  {cmidrule_header}
        {' & '.join([args.variables[o] for o in args.variable_order])} {metric_header} \\\midrule
{results_block}
        \bottomrule
    \end{{tabular}}
    \caption{{TODO: caption}}
    \label{{TODO: label}}
\end{{table}}"""

    print(latex_stub)
