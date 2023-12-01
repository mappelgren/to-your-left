import json
import logging
from abc import abstractmethod

import torch
from mlt.preexperiments.data_readers import DaleCaptionAttributeEncoder
from torch import nn
from torcheval.metrics import (
    BinaryAccuracy,
    Mean,
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MultilabelAccuracy,
)


class Tester:
    @abstractmethod
    def test(self, model, test_loader, device):
        pass


class DummyTester(Tester):
    def test(self, model, test_loader, device):
        return (json.dumps({"accuracy": 100}), [])


class AttentionPredictorTester(Tester):
    def test(self, model, test_loader, device):
        model.eval()
        # accuracy = MulticlassAccuracy(device=device)
        multilabel = MultilabelAccuracy(device=device)
        loss_function = nn.BCELoss()

        model_outputs = []
        ground_truths = []

        test_outputs = []
        for model_input, ground_truth, image_id in test_loader:
            model_input = [t.to(device) for t in model_input]
            ground_truth = ground_truth.to(device)

            output = model(model_input).detach()
            max_indices = torch.max(output, dim=1)[1]

            test_outputs.extend(zip(image_id, output, ground_truth))

            model_outputs.extend(output)
            ground_truths.extend(ground_truth)

            multilabel.update(output, ground_truth)
            # accuracy.update(output, ground_truth)

        loss = loss_function(torch.stack(model_outputs), torch.stack(ground_truths))
        return (
            json.dumps(
                {"accuracy": f"{multilabel.compute():.4f}", "loss": f"{loss:.4f}"}
            ),
            test_outputs,
        )


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

        return (
            json.dumps(
                {
                    "accuracy": f"{accuracy.compute():.4f}",
                    "mean test loss": f"{mean.compute():.4f}",
                }
            ),
            test_outputs,
        )


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

        return (
            json.dumps(
                {
                    "accuracy": f"{accuracy.compute():.4f}",
                }
            ),
            test_outputs,
        )


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
        class_precision = MulticlassPrecision(
            device=device,
            average=None,
            num_classes=len(DaleCaptionAttributeEncoder.vocab),
        )
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
            class_precision.update(output.flatten(), ground_truth.flatten())
            class_recall.update(output.flatten(), ground_truth.flatten())

            test_outputs.extend(zip(image_id, output, ground_truth))

        logging.getLogger().setLevel(logging.ERROR)
        computed_accuracy = word_by_word_accuracy.compute()
        computed_class_accuracy = class_accuracy.compute()
        computed_class_precisions = class_precision.compute()
        computed_class_recall = class_recall.compute()
        logging.getLogger().setLevel(logging.WARNING)

        accuracy_by_word = {
            word: round(accuracy.item(), 4)
            for word, accuracy in zip(
                DaleCaptionAttributeEncoder.vocab, computed_class_accuracy
            )
        }
        precision_by_word = {
            word: round(precision.item(), 4)
            for word, precision in zip(
                DaleCaptionAttributeEncoder.vocab, computed_class_precisions
            )
        }
        recall_by_word = {
            word: round(recall.item(), 4)
            for word, recall in zip(
                DaleCaptionAttributeEncoder.vocab, computed_class_recall
            )
        }

        included_indices = [
            i
            for i, token in enumerate(DaleCaptionAttributeEncoder.vocab)
            if token != DaleCaptionAttributeEncoder.SOS_TOKEN
        ]

        return (
            json.dumps(
                {
                    "accuracy": f"{accuracy.compute():.4f}",
                    "word_by_word_accuracy": f"{computed_accuracy:.4f}",
                    "accuracy_by_word": accuracy_by_word,
                    "word_by_word_precision": f"{torch.mean(computed_class_precisions[included_indices]):.4f}",
                    "precision_by_word": precision_by_word,
                    "word_by_word_recall": f"{torch.mean(computed_class_recall[included_indices]):.4f}",
                    "recall_by_word": recall_by_word,
                    "non_target_accuracy": f"{non_target_accuracy.compute():.4f}",
                }
            ),
            test_outputs,
        )
