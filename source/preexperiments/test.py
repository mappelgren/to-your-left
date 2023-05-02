from abc import abstractmethod

import torch
from torcheval.metrics import (
    BinaryAccuracy,
    Mean,
    MulticlassAccuracy,
    MultilabelAccuracy,
)


class Tester:
    @abstractmethod
    def test(self, model, test_loader, device):
        pass


class CoordinatePredictorTester(Tester):
    def test(self, model, test_loader, device):
        model.eval()
        accuracy = BinaryAccuracy(device=device)
        mean = Mean(device=device)

        test_outputs = []
        for model_input, ground_truth, image_id in test_loader:
            model_input = [t.to(device) for t in model_input]
            ground_truth = ground_truth.to(device)

            output = model(model_input).detach()
            test_outputs.extend(zip(image_id, output, ground_truth))

            distances = torch.diagonal(torch.cdist(output, ground_truth.float()))
            mean.update(distances)

            positives = torch.where(distances < 20, distances, 0)
            accuracy.update(positives, torch.ones_like(positives))

        return {
            "accuracy": f"{accuracy.compute():.2f}",
            "mean test loss": f"{mean.compute():.2f}",
        }, test_outputs


class BoundingBoxClassifierTester(Tester):
    def test(self, model, test_loader, device):
        model.eval()
        accuracy = MulticlassAccuracy(device=device)

        test_outputs = []
        for model_input, ground_truth, image_id in test_loader:
            model_input = model_input.to(device)
            ground_truth = ground_truth.to(device)
            output = model(model_input).detach()
            max_indices = torch.max(output, dim=1)[1]
            test_outputs.extend(zip(image_id, max_indices, ground_truth))

            accuracy.update(max_indices, ground_truth)

        return {
            "accuracy": f"{accuracy.compute():.2f}",
        }, test_outputs


class CaptionGeneratorTester(Tester):
    def test(self, model, test_loader, device):
        model.eval()
        accuracy = MultilabelAccuracy(device=device)
        hamming_accuracy = MultilabelAccuracy(criteria="hamming", device=device)
        non_target_accuracy = BinaryAccuracy(device=device)

        test_outputs = []
        for model_input, ground_truth, image_id in test_loader:
            image, _, non_target_captions, masked_image, *_ = model_input

            image = image.to(device)
            masked_image = masked_image.to(device)
            non_target_captions = non_target_captions.to(device)
            ground_truth = ground_truth.to(device)

            if masked_image.dim() != 1:
                output = model.caption(image, masked_image).detach()
            else:
                output = model.caption(image).detach()
            for sample, output_sample in zip(non_target_captions, output):
                for caption in sample:
                    described_non_target_object = torch.tensor(False)
                    if torch.equal(output_sample, caption):
                        described_non_target_object = torch.tensor(True)
                        break
                non_target_accuracy.update(
                    described_non_target_object.unsqueeze(dim=0),
                    torch.tensor(True).unsqueeze(dim=0),
                )

            test_outputs.extend(zip(image_id, output, ground_truth))

            print(output, ground_truth)

            accuracy.update(output, ground_truth)
            hamming_accuracy.update(output, ground_truth)

        return {
            "accuracy": f"{accuracy.compute():.2f}",
            "hamming_accuracy": f"{hamming_accuracy.compute():.2f}",
            "non_target_accuracy": f"{non_target_accuracy.compute():.2f}",
        }, test_outputs
