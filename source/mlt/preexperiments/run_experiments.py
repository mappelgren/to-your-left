import itertools
import subprocess

options = [
    "--dataset_base_dir=/home/dominik/Development/",
    "--epochs=30",
    "--lr=0.0002",
    "--feature_file=resnet_3_no-avgpool_no-fc.h5",
    "--max_samples=10000",
    "--device=cuda",
    "--model=attribute_coordinate_predictor",
]

variables = {
    "--dataset": ["single", "dale-2", "dale-5", "colour"],
    "--image_embedding_dimension": [100, 500, 1000],
    "--coordinate_classifier_dimension": [512, 1024, 2048],
}

for index, combination in enumerate(itertools.product(*variables.values())):
    if index == 0:
        continue
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
        ]
    )
