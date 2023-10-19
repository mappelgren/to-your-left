import itertools
import subprocess

options = [
    # "--dataset_base_dir=/scratch/guskunkdo",
    "--dataset_base_dir=/home/dominik/Development/",
    "--epochs=30",
    "--lr=0.0002",
    "--feature_file=resnet_3_no-avgpool_no-fc.h5",
    "--max_samples=10000",
    "--device=cuda",
    "--model=dale_attribute_coordinate_predictor",
    # "--out=/scratch/guskunkdo/out/",
]

variables = {
    "--dataset": ["dale-2", "dale-5", "colour"],
    "--image_embedding_dimension": [500, 1000],
    "--coordinate_classifier_dimension": [1024, 2048],
    # "--decoder_out_dim": [500, 1000, 1500, 2000],
    "--encoder_out_dim": [500, 1000, 1500, 2000],
    "--embedding_dim": [15, 30],
}

for index, combination in enumerate(itertools.product(*variables.values())):
    save_appendix = "_".join(str(i) for i in combination[1:])
    subprocess.run(
        [
            "python",
            "source/mlt/preexperiments/train.py",
            *options,
            *[
                f"{option}={value}"
                for option, value in zip(variables.keys(), combination)
            ],
            f"--save_appendix={save_appendix}",
        ],
        check=False,
    )
