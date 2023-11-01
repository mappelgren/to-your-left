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

    return run


def calculate_metrics(run):
    run = calculate_f1(run)

    return run


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
        "--datasets",
        nargs="+",
        type=str,
        help="datasets in order in which they should appear in the table",
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

    all_metrics = set(
        metric for experiment in experiments for metric in experiment.last_run.keys()
    )
    sorted_metrics = list(sorted(all_metrics))
    print(list(enumerate(sorted_metrics)))

    choice = input("choose important metrics (separated by comma):")
    chosen_metrics = [sorted_metrics[int(c)] for c in choice.split(",")]

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
                rf"{{{str(round(experiment.last_run[metric] * 100, 2)).rstrip('0.').replace('.', ',')}\%}}"
                for dataset in all_datasets
                for experiment in variation_experiments
                for metric in chosen_metrics
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
        {' & '.join([args.variables[o].replace('_', '-') for o in args.variable_order])} {metric_header} \\\midrule
{results_block}
        \bottomrule
    \end{{tabular}}
    \caption{{test}}
    \label{{test}}
\end{{table}}"""

    print(latex_stub)
