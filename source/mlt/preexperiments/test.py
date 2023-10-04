from abc import abstractmethod

import torch
from mlt.preexperiments.data_readers import DaleCaptionAttributeEncoder
from torcheval.metrics import (
    BinaryAccuracy,
    Mean,
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
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
            model_input = [t.to(device) for t in model_input]
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
        accuracy = BinaryAccuracy(device=device)
        word_by_word_accuracy = MulticlassAccuracy(device=device)
        class_accuracy = MulticlassAccuracy(
            device=device,
            average=None,
            num_classes=len(DaleCaptionAttributeEncoder.vocab),
        )
        word_by_word_precision = MulticlassPrecision(
            device=device,
        )
        class_precision = MulticlassPrecision(
            device=device,
            average=None,
            num_classes=len(DaleCaptionAttributeEncoder.vocab),
        )
        word_by_word_recall = MulticlassRecall(device=device)
        class_recall = MulticlassRecall(
            device=device,
            average=None,
            num_classes=len(DaleCaptionAttributeEncoder.vocab),
        )
        non_target_accuracy = BinaryAccuracy(device=device)

        test_outputs = []
        for model_input, ground_truth, image_id in test_loader:
            if len(model_input) > 3:
                image, _, non_target_captions, masked_image, *_ = model_input
            else:
                image, _, non_target_captions = model_input
                masked_image = torch.zeros(1)

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

            for output_sample, ground_truth_sample in zip(output, ground_truth):
                hit = torch.equal(output_sample, ground_truth_sample)

                accuracy.update(
                    torch.tensor(hit).unsqueeze(dim=0),
                    torch.tensor(True).unsqueeze(dim=0),
                )

            word_by_word_accuracy.update(output.flatten(), ground_truth.flatten())
            class_accuracy.update(output.flatten(), ground_truth.flatten())
            word_by_word_precision.update(output.flatten(), ground_truth.flatten())
            class_precision.update(output.flatten(), ground_truth.flatten())
            word_by_word_recall.update(output.flatten(), ground_truth.flatten())
            class_recall.update(output.flatten(), ground_truth.flatten())

            test_outputs.extend(zip(image_id, output, ground_truth))

        accuracy_by_word = {
            word: round(accuracy.item(), 2)
            for word, accuracy in zip(
                DaleCaptionAttributeEncoder.vocab, class_accuracy.compute()
            )
        }
        precision_by_word = {
            word: round(precision.item(), 2)
            for word, precision in zip(
                DaleCaptionAttributeEncoder.vocab, class_precision.compute()
            )
        }
        recall_by_word = {
            word: round(recall.item(), 2)
            for word, recall in zip(
                DaleCaptionAttributeEncoder.vocab, class_recall.compute()
            )
        }

        return {
            "accuracy": f"{accuracy.compute():.2f}",
            "word_by_word_accuracy": f"{word_by_word_accuracy.compute():.2f}",
            "accuracy_by_word": f"{accuracy_by_word}",
            "word_by_word_precision": f"{word_by_word_precision.compute():.2f}",
            "precision_by_word": f"{precision_by_word}",
            "word_by_word_recall": f"{word_by_word_recall.compute():.2f}",
            "recall_by_word": f"{recall_by_word}",
            "non_target_accuracy": f"{non_target_accuracy.compute():.2f}",
        }, test_outputs
