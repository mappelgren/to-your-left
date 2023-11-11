import itertools
import subprocess

options = [
    #"--dataset_base_dir=/scratch/guskunkdo",
    "--dataset_base_dir=/home/dominik/Development/",
    # "--lr=0.0002",
    "--epochs=40",
    "--feature_file=resnet_3_no-avgpool_no-fc.h5",
    "--max_samples=10000",
    "--device=cuda",
    "--model=dale_attribute_coordinate_predictor",
    "--out=out/",
    # "--out=/scratch/guskunkdo/out/",
]

variables = {
    #"--dataset": ["dale-2"],
    "--dataset": ["dale-5", "colour"],
    #"--image_embedding_dimension": [300, 500],
    "--image_embedding_dimension": [50, 100, 300, 500],
    "--coordinate_classifier_dimension": [1024, 2048],
    # "--decoder_out_dim": [100, 500, 1000],
    "--encoder_out_dim": [1000, 1500],
    "--embedding_dim": [15, 30],
    "--lr": [0.0001, 0.00005, 0.00003],
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
