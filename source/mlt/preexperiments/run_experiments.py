import itertools
import subprocess

options = [
    "--dataset_base_dir=/home/dominik/Development/",
    "--epochs=30",
    "--lr=0.002",
    "--feature_file=bounding_box_resnet_4_avgpool_no-fc.h5",
    "--max_samples=10000",
    "--device=cuda",
    "--model=bounding_box_caption_generator",
]

variables = {
    "--dataset": ["colour"],
    "--image_embedding_dimension": [100, 500, 1000],
    "--embedding_dim": [10, 15, 30],
    "--decoder_out_dim": [100, 500, 1000],
}

for combination in itertools.product(*variables.values()):
    subprocess.run(
        [
            "python",
            "source/mlt/preexperiments/train.py",
            *options,
            *[
                f"{option}={value}"
                for option, value in zip(variables.keys(), combination)
            ],
        ]
    )
