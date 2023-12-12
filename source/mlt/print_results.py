import argparse
import ast
import glob
import json
import os
import re

# necessary for input()
# pylint: disable-next=unused-import
import readline
from abc import abstractmethod
from dataclasses import dataclass

from tabulate import tabulate


class Run:
    def __init__(self, file) -> None:
        self.epochs = []
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("appendix: "):
                    self.variable_names = ast.literal_eval(
                        line.removeprefix("appendix: ")
                    )

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
                            try:
                                flattened[k] = float(v)
                            except ValueError:
                                continue

                    extended = self.calculate_metrics(flattened)

                    self.epochs.append(extended)

    def calculate_metrics(self, run):
        run = self._calculate_metrics_by_attribute(run)
        run = self._calculate_f1(run)
        return run

    def _calculate_f1(self, run):
        if "precision_by_word_<pad>" in run.keys():
            precision_prefix = "precision_by_word_"
            recall_prefix = "recall_by_word_"
        else:
            precision_prefix = "prec_"
            recall_prefix = "rec_"

        if (
            "word_by_word_precision" in run.keys()
            and "word_by_word_recall" in run.keys()
        ):
            run["word_by_word_f1"] = (
                2
                * (run["word_by_word_precision"] * run["word_by_word_recall"])
                / (run["word_by_word_precision"] + run["word_by_word_recall"])
            )

        if (
            f"{precision_prefix}<pad>" in run.keys()
            and f"{recall_prefix}<pad>" in run.keys()
        ):
            run["f1_<pad>"] = (
                2
                * (run[f"{precision_prefix}<pad>"] * run[f"{recall_prefix}<pad>"])
                / (run[f"{precision_prefix}<pad>"] + run[f"{recall_prefix}<pad>"])
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

    def _calculate_metrics_by_attribute(self, run):
        if "precision_by_word_<pad>" in run.keys():
            prefixes = {
                "precision": "precision_by_word_",
                "recall": "recall_by_word_",
                "accuracy": "accuracy_by_word_",
            }
        else:
            prefixes = {"precision": "prec_", "recall": "rec_", "accuracy": "acc_"}

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

        for metric, metric_prefix in prefixes.items():
            for attribute, values in attributes.items():
                for value in values:
                    if f"{metric_prefix}{value}" not in run.keys():
                        return run

                run[f"{metric}_{attribute}"] = sum(
                    run[f"{metric_prefix}{value}"] for value in values
                ) / len(values)

        return run

    @property
    def last_epoch(self):
        return self.epochs[-1]

    @property
    def metrics(self):
        return set(self.last_epoch.keys())


variable_dict = {
    "max_len": "$n$",
    "receiver_embedding": "$e_r$",
    "receiver_hidden": "$h_r$",
    "receiver_projection": "$p$",
    "receiver_image_embedding": "$e_{ri}$",
    "receiver_decoder_embedding": "$LSTM_e$",
    "receiver_decoder_out": "$LSTM_o$",
    "sender_embedding": "$e_s$",
    "sender_hidden": "$h_s$",
    "sender_image_embedding": "$e_{si}$",
    "vocab_size": "$|V|$",
}


@dataclass
class Experiment:
    folder_name: str
    dataset: str
    variables: dict
    run: Run


class Experiments:
    def __init__(self, root_folder, datasets) -> None:
        self.datasets = datasets
        self.chosen_variables = tuple()

        self.experiments: list[Experiment] = []
        for folder in glob.glob(root_folder):
            folder_name = folder.split("/")[-1]

            dataset, experiment_variables = self._extract_information_from_folder(
                folder_name
            )
            if dataset not in datasets:
                continue

            run = Run(os.path.join(folder, "log.txt"))

            if len(run.epochs) != 0:
                self.experiments.append(
                    Experiment(
                        folder_name=folder_name,
                        dataset=dataset,
                        variables={
                            variable_dict[variable_name]: variable
                            for variable_name, variable in zip(
                                run.variable_names, experiment_variables
                            )
                        },
                        run=run,
                    )
                )

    def _extract_information_from_folder(self, folder_name):
        pattern = r"_([^_]+)((_[\de\-\.]+)*)$"
        match = re.search(pattern, folder_name)
        if match:
            dataset = match.group(1)
            experiment_variables = [
                float(var) for var in match.group(2).split("_") if var != ""
            ]

        return dataset, experiment_variables

    def exclude_folders(self):
        excluded_folders = self._get_excluded_folders()

        if len(excluded_folders) != 0:
            excluded_indices = []
            for index in excluded_folders.split(","):
                if "-" in index:
                    range_indices = index.split("-")
                    excluded_indices.extend(
                        range(int(range_indices[0]), int(range_indices[1]) + 1)
                    )
                else:
                    excluded_indices.append(int(index))

            self.experiments = [
                experiment
                for index, experiment in enumerate(self.experiments)
                if index not in excluded_indices
            ]

    def _get_excluded_folders(self):
        for index, folder in enumerate(
            [experiment.folder_name for experiment in self.experiments]
        ):
            print(f"{index}: {folder}")
        return input("exclude folders (separated by comma, ranges with hyphen):")

    def sort_variables(self):
        all_variables = list(
            sorted(
                set(
                    variable
                    for experiment in self.experiments
                    for variable in experiment.variables
                )
            )
        )
        for index, variable in enumerate(all_variables):
            print(f"{index}: {variable}")
        chosen_variables = [
            int(index)
            for index in input(
                "choose variables and order (separated by comma):"
            ).split(",")
            if index != ""
        ]
        if len(chosen_variables) == 0:
            chosen_variables = list(range(len(all_variables)))

        self.chosen_variables = tuple(
            all_variables[variable_index] for variable_index in chosen_variables
        )

        for experiment in self.experiments:
            for variable_name in self.chosen_variables:
                if variable_name not in experiment.variables:
                    experiment.variables[variable_name] = float("nan")

    def __getitem__(self, index):
        return self.experiments[index]

    def __len__(self):
        return len(self.experiments)


class OutputProcessor:
    @classmethod
    def to_percent(cls, number, r=2):
        rounded = str(round(number * 100, r)).replace(".", ",").rstrip("0").rstrip(",")
        return rf"{rounded}%"

    @classmethod
    def to_number(cls, number, r=2):
        return rf"{str(round(number, r)).replace('.', ',').rstrip('0').rstrip(',')}"

    def choose_metrics(self, experiments):
        all_metrics = set(
            metric
            for experiment in experiments
            for metric in experiment.run.last_epoch.keys()
        )
        sorted_metrics = list(sorted(all_metrics))
        for index, metric in enumerate(sorted_metrics):
            print(f"{index}: {metric}")
        choice = input("choose important metrics (separated by comma):").split(",")

        metric_types = [
            metric_type
            for metric_type in input("choose metric types [%, num] (default %):").split(
                ","
            )
            if metric_type in METRIC_TYPES
        ]
        if len(metric_types) < len(choice):
            metric_types.extend(["%"] * (len(choice) - len(metric_types)))
        metric_types = metric_types[: len(choice)]

        return {
            sorted_metrics[int(c)]: METRIC_TYPES[metric_type]
            for c, metric_type in zip(choice, metric_types)
        }

    @abstractmethod
    def print(self):
        ...


class SortedOutputProcessor(OutputProcessor):
    def __init__(self, experiments: Experiments) -> None:
        self.experiments = experiments
        self.chosen_metrics = self.choose_metrics(experiments)

    def print(self):
        metric_sort_order = [
            order
            for order in input(
                "choose sort order for metrics [h, l] (default h):"
            ).split(",")
            if order in ["h", "l"]
        ]
        if len(metric_sort_order) < len(self.chosen_metrics):
            metric_sort_order.extend(
                ["h"] * (len(self.chosen_metrics) - len(metric_sort_order))
            )

        metric_sort_order = [
            1 if order == "h" else -1
            for order in metric_sort_order[: len(self.chosen_metrics)]
        ]

        sorted_experiments = sorted(
            self.experiments.experiments,
            key=lambda experiment: [
                experiment.run.last_epoch[metric] * metric_sort_order[index]
                for index, metric in enumerate(self.chosen_metrics)
            ],
            reverse=True,
        )
        headers = [
            "dataset",
            *self.experiments.chosen_variables,
            *list(self.chosen_metrics.keys()),
        ]
        rows = [
            [
                experiment.dataset,
                *[
                    experiment.variables[variable]
                    for variable in self.experiments.chosen_variables
                ],
                *[
                    metric_type(experiment.run.last_epoch[metric])
                    for metric, metric_type in self.chosen_metrics.items()
                ],
            ]
            for experiment in sorted_experiments
        ]
        print(tabulate(rows, headers, tablefmt="rounded_outline"))


class LatexOutputProcessor(OutputProcessor):
    def __init__(self, experiments: Experiments) -> None:
        self.experiments = experiments
        self.chosen_metrics = self.choose_metrics(experiments)

    def print(self):
        all_datasets = self.experiments.datasets
        all_variables = self.experiments.chosen_variables
        sorted_variations = sorted(
            set(
                tuple(
                    experiment.variables.get(variable, float("nan"))
                    for variable in self.experiments.chosen_variables
                )
                for experiment in self.experiments
            )
        )

        number_datasets = len(all_datasets)
        number_variables = len(all_variables)
        number_metrics = len(self.chosen_metrics)

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
            [rf"\textbf{{{metric}}}" for metric in self.chosen_metrics]
        )
        metric_header = f" & {metric_header_per_dataset}" * number_datasets

        result_lines = []
        for variation in sorted_variations:
            variation_experiments = [
                experiment
                for experiment in self.experiments.experiments
                if tuple(
                    experiment.variables[variable]
                    for variable in self.experiments.chosen_variables
                )
                == variation
            ]

            variable_string = "        " + " & ".join(
                [f"{{{self.to_number(variable, 10)}}}" for variable in variation]
            )

            result_string = " & ".join(
                [
                    rf"{{{metric_type(experiment.run.last_epoch[metric])}}}"
                    for dataset in all_datasets
                    for experiment in variation_experiments
                    for metric, metric_type in self.chosen_metrics.items()
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
        {' & '.join(self.experiments.chosen_variables)} {metric_header} \\\midrule
{results_block}
        \bottomrule
    \end{{tabular}}
    \caption{{TODO: caption}}
    \label{{TODO: label}}
\end{{table}}"""

        latex_stub = latex_stub.replace("%", r"\%").replace("_", r"\_")

        print(latex_stub)


METRIC_TYPES = {
    "%": OutputProcessor.to_percent,
    "num": OutputProcessor.to_number,
}
# fmt: off
PRINT_TYPES = {
    "latex": LatexOutputProcessor,
    "sorted": SortedOutputProcessor
}
# fmt: on

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="path to the results folders (can contain wildcards)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        help="datasets in order in which they should appear in the table",
    )
    parser.add_argument(
        "--output",
        choices=PRINT_TYPES.keys(),
        default="sorted",
        help="how the output should be printed",
    )

    args = parser.parse_args()

    print(args)

    all_experiments = Experiments(args.folder, args.datasets)
    all_experiments.exclude_folders()
    all_experiments.sort_variables()

    output_processor = PRINT_TYPES[args.output](all_experiments)
    output_processor.print()
