import argparse
import os
import re
from glob import glob

import torch

dataset_mapping = {
    "dale-2": "CLEVR_UNAMBIGOUS-DALE-TWO",
    "dale-5": "CLEVR_UNAMBIGOUS-DALE",
    "single": "CLEVR_RANDOM-SINGLE",
    "colour": "CLEVR_UNAMBIGOUS-COLOR",
}


def save_csv(run_dir, dataset, split, file_name):
    interaction_dir = os.path.join(run_dir, "interactions/")
    if not os.path.exists(interaction_dir):
        return

    with open(
        os.path.join(run_dir, f"{file_name}_outputs.csv"), "w", encoding="utf-8"
    ) as f:
        f.write("image_id,x,y,target_x,target_y\n")

    train_interaction = torch.load(
        glob(os.path.join(interaction_dir, f"{split}/", "epoch*/", "interaction*"))[0]
    )

    for image_id, label, receiver_output, length in zip(
        train_interaction.aux_input["image_id"],
        train_interaction.labels,
        train_interaction.receiver_output,
        train_interaction.message_length,
    ):
        final_output = receiver_output[int(length) - 1].tolist()
        label = label.tolist()
        image_name = f"{dataset_mapping[dataset]}_{str(int(image_id)).zfill(6)}"

        # attention predictor
        if len(final_output) > 2:
            final_output = str(
                [round(float(region), 4) for region in final_output]
            ).replace(",", ";")
            label = str([round(float(region), 4) for region in label]).replace(",", ";")
        # coordinate predictor
        else:
            final_output = f"{final_output[0]},{final_output[1]}"
            label = f"{label[0]},{label[1]}"

        with open(
            os.path.join(run_dir, f"{file_name}_outputs.csv"), "a", encoding="utf-8"
        ) as f:
            f.write(f"{image_name},{final_output},{label}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        help="Path to the root directory of all results",
    )

    args = parser.parse_args()

    for run in glob(os.path.join(args.root_dir, "*")):
        pattern = r"_([^_]+)((_[\de\-\.]+)*)$"
        match = re.search(pattern, run)
        if match:
            dataset = match.group(1)
            save_csv(run, dataset, "train", "train")
            save_csv(run, dataset, "validation", "test")
