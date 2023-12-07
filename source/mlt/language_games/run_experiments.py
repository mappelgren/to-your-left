import argparse
import itertools
import subprocess

options = [
    "--save",
    "--sender_cell=lstm",
    "--receiver_cell=lstm",
    "--image_feature_file=resnet_3_no-avgpool_no-fc.h5",
    "--bounding_box_feature_file=bounding_box_resnet_4_avgpool_no-fc.h5",
]

variables = {
    "--n_epochs": [100],
    "--batch_size": [32],
    "--batches_per_epoch": [40],
    "--validation_freq": [10],
    "--lr": [0.0002],
    "--max_samples": [10000],
    "--temperature": [1],
    "--model": ["caption_generator"],
    "--dataset": ["dale-2", "dale-5", "colour"],
    # -- model specifics --
    "--sender_image_embedding": [500],
    # "--sender_encoder_dim": [100],
    "--receiver_image_embedding": [100],
    "--receiver_decoder_out": [1500],
    "--receiver_embedding": [500],
    # "--receiver_coordinate_classifier_dimension": [100],
    "--sender_hidden": [100, 500],
    "--receiver_hidden": [10, 50, 500],
    "--vocab_size": [1, 10, 16, 50, 100],
    "--max_len": [1, 2, 3, 4, 6],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlt_server",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run on MLT server or not",
    )

    args, additional = parser.parse_known_args()

    if args.mlt_server:
        spec_options = [
            "--dataset_base_dir=/scratch/guskunkdo",
            "--out_dir=/scratch/guskunkdo/out/",
            "--validation_batch_size=1024",
            "--validation_batches_per_epoch=2",
        ]
    else:
        spec_options = [
            "--dataset_base_dir=/home/dominik/Development/",
            "--out_dir=out/",
            "--validation_batch_size=256",
            "--validation_batches_per_epoch=8",
        ]

    for index, combination in enumerate(itertools.product(*variables.values())):
        subprocess.run(
            [
                "python",
                "source/mlt/language_games/play.py",
                *spec_options,
                *options,
                *[
                    f"{option}={value}"
                    for option, value in zip(variables.keys(), combination)
                ],
                *additional,
            ],
            check=False,
        )
