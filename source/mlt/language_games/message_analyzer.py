import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from glob import glob

import numpy as np
import torch
from egg.core import Interaction
from mlt.preexperiments.data_readers import Color, Shape, Size
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


@dataclass
class LanguageTranslationSample:
    image_id: str
    emerged_message: tuple
    english_description: tuple


class LanguageTranslationDataset(Dataset):
    dataset_mapping = {
        "dale-2": ("CLEVR_UNAMBIGOUS-DALE-TWO", "clevr-images-unambigous-dale-two"),
        "dale-5": ("CLEVR_UNAMBIGOUS-DALE", "clevr-images-unambigous-dale"),
        "single": ("CLEVR_RANDOM-SINGLE", "clevr-images-random-single"),
        "colour": ("CLEVR_UNAMBIGOUS-COLOR", "clevr-images-unambigous-colour"),
    }

    combinations = [
        ("shape", "color", "size"),
        ("shape", "size", "color"),
        ("color", "shape", "size"),
        ("color", "size", "shape"),
        ("size", "shape", "color"),
        ("size", "color", "shape"),
    ]

    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"

    # class variable, because vocab is static
    english_vocab = {
        word: index
        for index, word in enumerate(
            list(
                [
                    PAD_TOKEN,
                    SOS_TOKEN,
                    EOS_TOKEN,
                    *[
                        word.lower()
                        for word in [*Size.names(), *Color.names(), *Shape.names()]
                    ],
                ]
            )
        )
    }

    def __init__(
        self, emerged_vocab_size, dataset, interaction_path, scenes_root, combination
    ) -> None:
        self.emerged_vocab_size = emerged_vocab_size
        self.samples: list[LanguageTranslationSample] = []

        interaction: Interaction = torch.load(interaction_path)

        for raw_message, image_id in zip(
            interaction.message, interaction.aux_input["image_id"]
        ):
            image_id = self._get_image_name(image_id, dataset)
            message = self._remove_eos(raw_message.max(dim=1).indices)
            english_description = self._get_english_description(
                scenes_root, image_id, combination
            )

            self.samples.append(
                LanguageTranslationSample(
                    image_id=image_id,
                    emerged_message=torch.tensor(message),
                    english_description=torch.tensor(
                        [self.english_vocab[word] for word in english_description]
                    ),
                )
            )

    def _get_image_name(self, image_id, dataset):
        return f"{self.dataset_mapping[dataset][0]}_{str(int(image_id)).zfill(6)}"

    def _remove_eos(self, tensor):
        for index, symbol in enumerate(tensor):
            if int(symbol) == 0:
                return tuple(tensor[:index].tolist())

    def _get_english_description(self, scenes_root, image_id, combination):
        scene_file = os.path.join(scenes_root, f"{image_id}.json")

        with open(scene_file, "r", encoding="utf-8") as f:
            scene = json.load(f)

        image_id = scene_file.split("/")[-1].removesuffix(".json")
        target_object_index = scene["groups"]["target"][0]
        target_object = scene["objects"][target_object_index]

        description = self._get_dale(
            combination,
            (
                target_object[combination[0]],
                target_object[combination[1]],
                target_object[combination[2]],
            ),
            scene,
        )

        return [self.SOS_TOKEN, *description, self.EOS_TOKEN]

    def _get_dale(self, order, target_attributes, scene):
        caption = [target_attributes[0]]
        remaining_objects = [
            obj for obj in scene["objects"] if obj[order[0]] == target_attributes[0]
        ]

        if len(remaining_objects) > 1:
            caption.insert(0, target_attributes[1])
            remaining_objects = [
                obj
                for obj in remaining_objects
                if obj[order[1]] == target_attributes[1]
            ]

            if len(remaining_objects) > 1:
                caption.insert(0, target_attributes[2])

        return tuple(caption)

    def __getitem__(self, index):
        sample = self.samples[index]

        return (
            (
                sample.emerged_message,
                sample.english_description,
            ),
            sample.english_description,
            sample.image_id,
        )

    def __len__(self):
        return len(self.samples)


class LanguageEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, encoding_size) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, encoding_size, batch_first=True)

    def forward(self, description):
        embedded = self.embedding(description)
        _, states = self.lstm(embedded)

        return states


class LanguageDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, decoding_size) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, decoding_size, batch_first=True)
        self.classifier = nn.LazyLinear(vocab_size)

    def forward(self, word, hidden_state):
        embedded = self.embedding(word)
        output, hidden_state = self.lstm(embedded, hidden_state)
        classified = self.classifier(output).squeeze(1)

        return classified, hidden_state


class BaselineLanguageTranslationModel(nn.Module):
    def __init__(
        self,
        language_encoder,
        translation_space,
        language_decoder,
    ) -> None:
        super().__init__()

        self.language_encoder = language_encoder
        self.hidden = nn.LazyLinear(translation_space)
        self.language_decoder = language_decoder

    def forward(self, data):
        _, description, *_ = data
        predicted_output = []

        encoded_states = self.language_encoder(
            torch.zeros((description.shape[0], 1)).long()
        )
        state = self.hidden(encoded_states[0])
        states = (state, state)

        target = description.transpose(0, 1)
        for word in target[:-1]:
            output, states = self.language_decoder(word.unsqueeze(1), states)
            predicted_output.append(output)

        return torch.stack(predicted_output).permute(1, 2, 0)


class EnglishLanguageTranslationModel(nn.Module):
    def __init__(
        self,
        language_encoder,
        translation_space,
        language_decoder,
    ) -> None:
        super().__init__()

        self.language_encoder = language_encoder
        self.hidden = nn.LazyLinear(translation_space)
        self.language_decoder = language_decoder

    def forward(self, data):
        _, description, *_ = data
        predicted_output = []

        encoded_states = self.language_encoder(description)
        state = self.hidden(encoded_states[0])
        states = (state, state)

        target = description.transpose(0, 1)
        for word in target[:-1]:
            output, states = self.language_decoder(word.unsqueeze(1), states)
            predicted_output.append(output)

        return torch.stack(predicted_output).permute(1, 2, 0)


class EmergentLanguageTranslationModel(nn.Module):
    def __init__(
        self,
        language_encoder,
        translation_space,
        language_decoder,
    ) -> None:
        super().__init__()

        self.language_encoder = language_encoder
        self.hidden = nn.LazyLinear(translation_space)
        self.language_decoder = language_decoder

    def forward(self, data):
        emerged, description, *_ = data
        predicted_output = []

        encoded_states = self.language_encoder(emerged)
        state = self.hidden(encoded_states[0])
        states = (state, state)

        target = description.transpose(0, 1)
        for word in target[:-1]:
            output, states = self.language_decoder(word.unsqueeze(1), states)
            predicted_output.append(output)

        return torch.stack(predicted_output).permute(1, 2, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # -- DATASET --
    parser.add_argument(
        "--run_root_dir", type=str, help="Path to the root dir of the runs"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset (dale-2, dale-5, single, colour)",
    )
    parser.add_argument(
        "--run_folder",
        type=str,
        default=None,
        help="Name of the run",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the model (discriminator, ...)",
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default=None,
        help="Path to the root of the datasets",
    )
    parser.add_argument(
        "--emerged_vocab_size",
        type=int,
        default=None,
        help="Size of the emerged language",
    )
    parser.add_argument(
        "--combination",
        type=int,
        default=0,
        help="Number of combination",
    )
    parser.add_argument(
        "--translator",
        choices=["english", "baseline", "emergent"],
        help="Translator to load",
    )

    # -- TRAINING --
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")

    args = parser.parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        raise AttributeError("Device must be cpu or cuda")

    model_dir = os.path.join(args.run_root_dir, args.model_name)
    dataset_dir = os.path.join(model_dir, f"{args.dataset_name}/")
    run_dir = os.path.join(dataset_dir, args.run_folder)
    train_interaction_path = glob(
        os.path.join(run_dir, "interactions/", "train/", "epoch*/", "interaction*")
    )[0]
    test_interaction_path = glob(
        os.path.join(run_dir, "interactions/", "validation/", "epoch*/", "interaction*")
    )[0]

    scenes_root = os.path.join(
        args.dataset_root_dir,
        LanguageTranslationDataset.dataset_mapping[args.dataset_name][1],
        "scenes/",
    )

    dataset = LanguageTranslationDataset(
        dataset=args.dataset_name,
        interaction_path=test_interaction_path,
        scenes_root=scenes_root,
        combination=LanguageTranslationDataset.combinations[args.combination],
        emerged_vocab_size=args.emerged_vocab_size,
    )

    language_decoder = LanguageDecoder(len(dataset.english_vocab), 10, 10)

    if args.translator == "baseline":
        translator = BaselineLanguageTranslationModel(
            language_encoder=LanguageEncoder(1, 10, 10),
            translation_space=10,
            language_decoder=language_decoder,
        )
    elif args.translator == "english":
        translator = EnglishLanguageTranslationModel(
            language_encoder=LanguageEncoder(len(dataset.english_vocab), 10, 10),
            translation_space=10,
            language_decoder=language_decoder,
        )
    elif args.translator == "emergent":
        translator = EmergentLanguageTranslationModel(
            language_encoder=LanguageEncoder(dataset.emerged_vocab_size, 10, 10),
            translation_space=10,
            language_decoder=language_decoder,
        )

    def collate(data):
        sources = []
        targets = []
        outputs = []
        image_ids = []

        for (source, target), output, image_id in data:
            sources.append(torch.cat((source, torch.zeros(3 - source.shape[0]))))
            targets.append(torch.cat((target, torch.zeros(5 - target.shape[0]))))
            outputs.append(torch.cat((output, torch.zeros(5 - output.shape[0]))))
            image_ids.append(image_id)

        return (
            (torch.stack(sources), torch.stack(targets)),
            torch.stack(outputs),
            image_ids,
        )

    loss_function = nn.CrossEntropyLoss()
    train_dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    optimizer = optim.Adam(translator.parameters(), lr=args.lr)

    print(
        f"{args.epochs} EPOCHS - {math.floor(len(dataset) / train_dataloader.batch_size)} BATCHES PER EPOCH"
    )

    for epoch in range(args.epochs):
        total_loss = 0
        for i, (model_input, target, image_ids) in enumerate(train_dataloader):
            model_input = [tensor.to(device).long() for tensor in model_input]
            target = target.to(device).long()

            output = translator(model_input).float()

            # print(list(zip(target, output.max(dim=1).indices)))

            loss = loss_function(
                output,
                target[:, 1:],
            )
            total_loss += loss.item()

            # print average loss for the epoch
            sys.stdout.write(
                f"\repoch {epoch}, batch {i}: {np.round(total_loss / (i + 1), 4)}"
            )

            # compute gradients
            loss.backward()
            # update parameters
            optimizer.step()
            # reset gradients
            optimizer.zero_grad()
        print()
