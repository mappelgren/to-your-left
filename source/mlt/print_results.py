import argparse
import glob
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folder",
        type=str,
        help="path to the results folders (can contain wildcards)",
    )

    args = parser.parse_args()
    print(args)

    metrics = set()
    all_results = {}
    for folder in glob.glob(args.folder):
        name = folder.split("/")[-1]
        with open(os.path.join(folder, "log.txt"), "r", encoding="utf-8") as f:
            results = []
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

                    results.append(flattened)

            if len(results) != 0:
                metrics.update(results[0].keys())
                all_results[name] = results

    sorted_metrics = list(sorted(metrics))
    print(list(enumerate(sorted_metrics)))

    choice = input("choose important metrics (separate by comma):")
    chosen_metrics = [sorted_metrics[int(c)] for c in choice.split(",")]

    last_runs = {run_name: runs[-1] for run_name, runs in all_results.items()}

    sorted_runs = sorted(
        last_runs.items(),
        key=lambda x: [x[1][metric] for metric in chosen_metrics],
        reverse=True,
    )

    print(chosen_metrics)
    for run_name, run_metrics in sorted_runs:
        print(run_name, [run_metrics[metric] for metric in chosen_metrics])
