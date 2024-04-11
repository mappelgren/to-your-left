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
    def __init__(self, file, dataset, baseline_runs=None, english_runs=None) -> None:
        self.epochs = []
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("('"):
                    self.combination = ast.literal_eval(line)

                if line.startswith("epoch"):
                    metrics = {"loss": float(line.split()[-1])}

                    if baseline_runs is not None and english_runs is not None:
                        baseline = [
                            baseline_run
                            for baseline_run in baseline_runs[dataset]
                            if baseline_run.combination == self.combination
                        ][0].last_epoch
                        english = [
                            english_run
                            for english_run in english_runs[dataset]
                            if english_run.combination == self.combination
                        ][0].last_epoch

                        metrics["normalized_loss"] = (
                            metrics["loss"] - english["loss"]
                        ) / (baseline["loss"] - english["loss"])

                    self.epochs.append(metrics)

    @property
    def last_epoch(self):
        return self.epochs[-1]

    @property
    def metrics(self):
        return set(self.last_epoch.keys())


variable_dict = {
    "max_len": "$n$",
    "encoder_embedding": "$LSTM_e$",
    "encoder_out": "$LSTM_o$",
    "projection": "$p$",
    "receiver_embedding": "$e_r$",
    "receiver_hidden": "$h_r$",
    "receiver_projection": "$p$",
    "receiver_image_embedding": "$e_{ri}$",
    "receiver_decoder_embedding": "$LSTM_e$",
    "receiver_decoder_out": "$LSTM_o$",
    "receiver_coordinate_classifier_dimension": "$c$",
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
    def __init__(self, root_folder, datasets, baseline_runs, english_runs) -> None:
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

            with open(os.path.join(folder, "log.txt"), "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("appendix: "):
                        variable_names = ast.literal_eval(
                            line.removeprefix("appendix: ")
                        )
                        break

            for combination_folder in glob.glob(os.path.join(folder, "translate*")):
                run = Run(combination_folder, dataset, baseline_runs, english_runs)
                if len(run.epochs) > 0:
                    self.experiments.append(
                        Experiment(
                            folder_name=folder_name,
                            dataset=dataset,
                            variables={
                                variable_dict[variable_name]: variable
                                for variable_name, variable in zip(
                                    variable_names, experiment_variables
                                )
                            },
                            run=run,
                        )
                    )

            self.combinations = sorted(
                set(experiment.run.combination for experiment in self.experiments)
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

    def filter_variables(self, variables):
        renamed_variables = {
            variable_dict[key]: float(value) for key, value in variables.items()
        }

        self.experiments = [
            experiment
            for experiment in self.experiments
            if set(renamed_variables.items()).issubset(
                set(experiment.variables.items())
            )
        ]

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

    @abstractmethod
    def print(self): ...


class SortedOutputProcessor(OutputProcessor):
    def __init__(self, experiments: Experiments) -> None:
        self.experiments = experiments
        self.chosen_metrics = {"normalized_loss": METRIC_TYPES["%"]}

    def print(self):
        sorted_experiments = sorted(
            self.experiments.experiments,
            key=lambda experiment: [
                experiment.run.last_epoch[metric] for metric in self.chosen_metrics
            ],
            reverse=False,
        )
        headers = [
            "dataset",
            *self.experiments.chosen_variables,
            "combination",
            *list(self.chosen_metrics.keys()),
        ]
        rows = [
            [
                experiment.dataset,
                *[
                    experiment.variables[variable]
                    for variable in self.experiments.chosen_variables
                ],
                experiment.run.combination,
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
        self.chosen_metrics = {"normalized_loss": METRIC_TYPES["%"]}

    def print(self):
        all_variables = self.experiments.chosen_variables
        all_combinations = self.experiments.combinations
        sorted_variations = sorted(
            set(
                tuple(
                    experiment.variables.get(variable, float("nan"))
                    for variable in self.experiments.chosen_variables
                )
                for experiment in self.experiments
            )
        )

        number_variables = len(all_variables)
        number_metrics = len(self.chosen_metrics)
        number_combinations = len(all_combinations)

        combination_header = " & ".join(
            [
                rf"\textbf{{{self.format_combination(combination)}}}"
                for combination in all_combinations
            ]
        )

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
                    for combination in all_combinations
                    for experiment in variation_experiments
                    for metric, metric_type in self.chosen_metrics.items()
                    if experiment.run.combination == combination
                ]
            )
            result_lines.append(rf"{variable_string} & {result_string} \\")

        results_block = "\n".join(result_lines)

        latex_stub = rf"""\begin{{table}}[ht]
    \centering
    \begin{{tabular}}{{{'c'*number_variables}{f"|{'c'*number_metrics}"*number_combinations}}}
        \toprule
        {' & '.join(self.experiments.chosen_variables)} & {combination_header} \\\midrule
{results_block}
        \bottomrule
    \end{{tabular}}
    \caption{{TODO: caption}}
    \label{{TODO: label}}
\end{{table}}"""

        latex_stub = latex_stub.replace("%", r"\%").replace("_", r"\_")

        print(latex_stub)

    def format_combination(self, combination):
        abbr = {"color": "C", "shape": "Sh", "size": "Si"}
        return f"{' > '.join([abbr[attribute] for attribute in combination])}"


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


class NotABaselineFolder(Exception):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="path to the results folders (can contain wildcards)",
    )
    parser.add_argument(
        "--baseline_folder",
        type=str,
        help="path to the folders that contain runs for baseline and english translations",
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

    args, additional_args = parser.parse_known_args()
    print(args)
    additional_args = {
        key.removeprefix("--"): value
        for key, value in zip(additional_args[::2], additional_args[1::2])
    }
    print(additional_args)

    baseline_runs = {}
    english_runs = {}
    for dataset in args.datasets:
        dataset_folder = os.path.join(args.baseline_folder, dataset)
        if not os.path.exists(dataset_folder):
            raise NotABaselineFolder(
                f"Folder '{args.baseline_folder}' doesn't contain dataset '{dataset}'"
            )

        baseline_runs[dataset] = [
            Run(file, dataset)
            for file in glob.glob(os.path.join(dataset_folder, "*baseline*"))
        ]
        english_runs[dataset] = [
            Run(file, dataset)
            for file in glob.glob(os.path.join(dataset_folder, "*english*"))
        ]

    all_experiments = Experiments(
        args.folder, args.datasets, baseline_runs, english_runs
    )
    all_experiments.filter_variables(additional_args)
    all_experiments.exclude_folders()
    all_experiments.sort_variables()

    output_processor = PRINT_TYPES[args.output](all_experiments)
    output_processor.print()
