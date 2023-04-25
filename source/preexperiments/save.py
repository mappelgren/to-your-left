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


class BoundingBoxOutputProcessor(StandardOutputProcessor):
    def process(self, outputs):
        processed_output = [self.output_fields]

        for image_id, output, ground_truth in outputs:
            # test data
            if output.dim() == 0:
                index = output

            # train data
            else:
                index = torch.max(output, dim=0).indices

            processed_output.append((image_id, str(int(index)), str(int(ground_truth))))

        return processed_output


class PixelOutputProcessor(StandardOutputProcessor):
    def process(self, outputs):
        processed_output = [self.output_fields]
        for image_id, pixels, ground_truth in outputs:
            string_pixels = [str(p) for p in pixels.tolist()]
            string_ground_truths = [str(p) for p in ground_truth.tolist()]
            processed_output.append((image_id, *string_pixels, *string_ground_truths))

        return processed_output


class CaptionOutputProcessor(StandardOutputProcessor):
    def process(self, outputs):
        processed_output = [self.output_fields]
        for image_id, output, ground_truth in outputs:
            # test data
            if output.dim() == 1:
                encoded_caption = output

            # train data
            elif output.dim() == 2:
                encoded_caption = torch.max(output, dim=0).indices

            decoded_caption = " ".join(
                [self.dataset.get_decoded_word(index) for index in encoded_caption]
            )
            decoded_target_caption = " ".join(
                [self.dataset.get_decoded_word(index) for index in ground_truth]
            )
            processed_output.append((image_id, decoded_caption, decoded_target_caption))

        return processed_output
