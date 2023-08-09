import os
from glob import glob

import torch

root_dir = "/home/dominik/Nextcloud/020_Masterstudium/Language Technology/LT2402_Master Thesis/experiments/runs/language-games/masked_coordinate_predictor/"

dataset_mapping = {
    "dale-2": "CLEVR_UNAMBIGOUS-DALE-TWO",
    "dale-5": "CLEVR_UNAMBIGOUS-DALE",
    "single": "CLEVR_RANDOM-SINGLE",
    "colour": "CLEVR_UNAMBIGOUS-COLOR",
}


def save_csv(run_dir, dataset, split, file_name):
    with open(
        os.path.join(run_dir, f"{file_name}_outputs.csv"), "w", encoding="utf-8"
    ) as f:
        f.write("image_id,x,y,target_x,target_y\n")

    train_interaction = torch.load(
        glob(
            os.path.join(
                run_dir, "interactions/", f"{split}/", "epoch*/", "interaction*"
            )
        )[0]
    )

    for image_id, label, receiver_output, length in zip(
        train_interaction.aux_input["image_id"],
        train_interaction.labels,
        train_interaction.receiver_output,
        train_interaction.aux["length"],
    ):
        final_output = receiver_output[int(length) - 1].tolist()
        label = label.tolist()
        image_name = f"{dataset_mapping[dataset]}_{str(int(image_id)).zfill(6)}"

        with open(
            os.path.join(run_dir, f"{file_name}_outputs.csv"), "a", encoding="utf-8"
        ) as f:
            f.write(
                f"{image_name},{final_output[0]},{final_output[1]},{label[0]},{label[1]}\n"
            )


if __name__ == "__main__":
    for dataset_folder in glob(os.path.join(root_dir, "*")):
        dataset = dataset_folder.split("/")[-1]
        for run in glob(os.path.join(dataset_folder, "*")):
            save_csv(run, dataset, "train", "train")
            save_csv(run, dataset, "validation", "test")
