import argparse
import itertools
import subprocess

options = [
    "--n_epochs=30",
    "--validation_freq=30",
    "--batch_size=32",
    "--batches_per_epoch=10",
    "--lr=0.0002",
    "--feature_file=resnet_3_no-avgpool_no-fc.h5",
    "--max_samples=10000",
    "--model=discriminator",
    "--mode=gs",
    "--sender_cell=lstm",
    "--receiver_cell=lstm",
    "--save",
    # -- model specifics --
    "--sender_embedding=500",
    # "--sender_encoder_dim=",
    "--receiver_embedding=100",
    # "--receiver_decoder_out_dim=",
    # "--coordinate_classifier_dimension=",
    "--temperature=1",
]

variables = {
    "--dataset": ["dale-2", "dale-5", "colour"],
    "--sender_hidden": [100, 500, 1000],
    "--receiver_hidden": [10, 30, 50, 100, 500, 1000],
    "--vocab_size": [1, 10, 16, 50, 100],
    "--max_len": [1, 2, 3, 4, 5, 6],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlt_server",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run on MLT server or not",
    )

    args = parser.parse_args()

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
        save_appendix = "_".join(str(i) for i in combination[1:])
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
                f"--save_appendix={save_appendix}",
            ],
            check=False,
        )
