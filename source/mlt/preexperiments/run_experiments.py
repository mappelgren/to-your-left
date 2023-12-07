import itertools
import subprocess

variables = {
    # "--dataset_base_dir": ["/scratch/guskunkdo"],
    "--dataset_base_dir": ["/home/dominik/Development/"],
    "--lr": ["0.0002"],
    "--epochs": ["30"],
    "--image_feature_file": ["resnet_3_no-avgpool_no-fc.h5"],
    "--max_samples": ["10000"],
    "--device": ["cuda"],
    "--model": ["dale_attribute_attention_predictor"],
    "--out": ["out/"],
    # "--out": ["/scratch/guskunkdo/out/"],
    "--dataset": ["dale-2", "dale-5", "colour"],
    # "--image_embedding_dimension": [50, 100, 300, 500],
    # "--coordinate_classifier_dimension": [1024, 2048],
    # "--decoder_out_dim": [2000],
    "--encoder_out": [1500],
    "--encoder_embedding": [15],
    "--projection": [50, 100, 300, 500, 1000],
}

for index, combination in enumerate(itertools.product(*variables.values())):
    subprocess.run(
        [
            "python",
            "source/mlt/preexperiments/train.py",
            *[
                f"{option}={value}"
                for option, value in zip(variables.keys(), combination)
            ],
        ],
        check=False,
    )
