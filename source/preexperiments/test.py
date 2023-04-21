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
            test_outputs.extend(zip(image_id, output))

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
        accuracy = MulticlassAccuracy()

        test_outputs = []
        for model_input, ground_truth, image_id in test_loader:
            model_input = model_input.to(device)
            ground_truth = ground_truth.to(device)
            output = model(model_input).detach()
            max_indices = torch.max(output, dim=1)[1]
            test_outputs.extend(zip(image_id, max_indices))

            accuracy.update(max_indices, ground_truth)

        return {
            "accuracy": f"{accuracy.compute():.2f}",
        }, test_outputs


class CaptionGeneratorTester(Tester):
    def test(self, model, test_loader, device):
        model.eval()
        accuracy = MultilabelAccuracy()

        test_outputs = []
        for model_input, ground_truth, image_id in test_loader:
            model_input = model_input[0].to(device)
            ground_truth = ground_truth.to(device)
            output = model.caption(model_input).detach()

            test_outputs.extend(zip(image_id, output))
            print(output.device, ground_truth.device)
            accuracy.update(output, ground_truth)

        return {
            "accuracy": f"{accuracy.compute():.2f}",
        }, test_outputs
