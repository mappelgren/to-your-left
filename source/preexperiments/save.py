import os
from time import gmtime, strftime

import torch


class ModelSaver:
    def __init__(self, out_dir, model_name, output_processor) -> None:
        folder_name = f'{strftime("%Y-%m-%d_%H-%M-%S", gmtime())}_{model_name}'
        self.directory = os.path.join(out_dir, folder_name)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.output_processor = output_processor

    def save_model(self, model, file_name):
        torch.save(
            model.state_dict(),
            os.path.join(self.directory, file_name),
        )

    def save_log(self, log, file_name):
        with open(os.path.join(self.directory, file_name), "w", encoding="utf-8") as f:
            f.writelines(log)

    def save_output(self, output, file_name):
        processed_output = self.output_processor.process(output)
        self._save_to_csv(
            processed_output,
            os.path.join(self.directory, file_name),
        )

    def _save_to_csv(self, data, file):
        with open(file, "w", encoding="utf-8") as f:
            for line in data:
                f.write(",".join(line) + "\n")


class StandardOutputProcessor:
    def __init__(self, output_fields, dataset) -> None:
        self.output_fields = output_fields
        self.dataset = dataset

    def process(self, outputs):
        return [self.output_fields, *outputs]


class PixelOutputProcessor(StandardOutputProcessor):
    def process(self, outputs):
        processed_output = [self.output_fields]
        for image_id, pixels in outputs:
            string_pixels = [str(p) for p in pixels.tolist()]
            processed_output.append((image_id, *string_pixels))

        return processed_output


class CaptionOutputProcessor(StandardOutputProcessor):
    def process(self, outputs):
        processed_output = [self.output_fields]
        for image_id, output in outputs:
            # test data
            if output.dim() == 1:
                encoded_caption = output

            # train data
            elif output.dim() == 2:
                encoded_caption = torch.max(output, dim=0).indices

            decoded_caption = " ".join(
                [self.dataset.get_decoded_word(index) for index in encoded_caption]
            )
            processed_output.append((image_id, decoded_caption))

        return processed_output
