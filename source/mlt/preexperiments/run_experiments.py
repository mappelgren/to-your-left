import itertools
import subprocess

options = [
    "--dataset_base_dir=/home/dominik/Development/",
    "--epochs=30",
    "--lr=0.002",
    "--feature_file=resnet_3_no-avgpool_no-fc.h5",
    "--max_samples=10000",
    "--device=cuda",
    "--model=masked_caption_generator",
]

variables = {
    "--dataset": ["dale-2", "dale-5", "colour"],
    "--embedding_dim": [10, 15, 30],
    "--decoder_out_dim": [100, 500, 1000],
}

for combination in itertools.product(*variables.values()):
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
