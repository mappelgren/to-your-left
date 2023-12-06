import logging

import torch
import torch.nn.functional as F
from mlt.preexperiments.data_readers import DaleCaptionAttributeEncoder
from torcheval.metrics import (
    BinaryAccuracy,
    Mean,
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
)


def classification_loss(
    _sender_input,
    _message,
    _receiver_input,
    receiver_output,
    labels,
    _aux_input,
):
    # in the discriminative case, accuracy is computed by comparing the index with highest score in Receiver output (a distribution of unnormalized
    # probabilities over target poisitions) and the corresponding label read from input, indicating the ground-truth position of the target
    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    # similarly, the loss computes cross-entropy between the Receiver-produced target-position probability distribution and the labels
    loss = F.nll_loss(receiver_output, labels, reduction="none")
    return loss, {"acc": acc}


def captioning_loss(
    _sender_input,
    _message,
    _receiver_input,
    receiver_output,
    labels,
    aux_input,
):
    device = receiver_output.device
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

    predicted_captions = receiver_output.max(dim=1).indices
    for sample, output_sample in zip(
        aux_input["non_target_captions"], predicted_captions
    ):
        for caption in sample:
            described_non_target_object = torch.tensor(False)
            if torch.equal(output_sample, caption[1:]):
                described_non_target_object = torch.tensor(True)
                break
        non_target_accuracy.update(
            described_non_target_object.unsqueeze(dim=0),
            torch.tensor(True).unsqueeze(dim=0),
        )

    for output_sample, ground_truth_sample in zip(predicted_captions, labels):
        hit = torch.equal(output_sample, ground_truth_sample)

        accuracy.update(
            torch.tensor(hit).unsqueeze(dim=0),
            torch.tensor(True).unsqueeze(dim=0),
        )

    word_by_word_accuracy.update(predicted_captions.flatten(), labels.flatten())
    class_accuracy.update(predicted_captions.flatten(), labels.flatten())
    class_precision.update(predicted_captions.flatten(), labels.flatten())
    class_recall.update(predicted_captions.flatten(), labels.flatten())

    logging.getLogger().setLevel(logging.ERROR)
    computed_class_accuracy = class_accuracy.compute()
    computed_class_precisions = class_precision.compute()
    computed_class_recall = class_recall.compute()
    logging.getLogger().setLevel(logging.WARNING)

    accuracy_by_word = {
        f"acc_{word}": accuracy.detach().clone().float()
        for word, accuracy in zip(
            DaleCaptionAttributeEncoder.vocab, computed_class_accuracy
        )
    }
    precision_by_word = {
        f"prec_{word}": precision.detach().clone().float()
        for word, precision in zip(
            DaleCaptionAttributeEncoder.vocab, computed_class_precisions
        )
    }
    recall_by_word = {
        f"rec_{word}": recall.detach().clone().float()
        for word, recall in zip(
            DaleCaptionAttributeEncoder.vocab, computed_class_recall
        )
    }

    loss = F.cross_entropy(receiver_output, labels)

    included_indices = [
        i
        for i, token in enumerate(DaleCaptionAttributeEncoder.vocab)
        if token != DaleCaptionAttributeEncoder.SOS_TOKEN
    ]

    return loss, {
        "accuracy": accuracy.compute().detach().clone().float(),
        "word-by-word_accuracy": word_by_word_accuracy.compute()
        .detach()
        .clone()
        .float(),
        "non_target_accuracy": non_target_accuracy.compute().detach().clone().float(),
        "word_by_word_precision": torch.mean(
            computed_class_precisions[included_indices]
        ),
        "word_by_word_recall": torch.mean(computed_class_recall[included_indices]),
        **accuracy_by_word,
        **precision_by_word,
        **recall_by_word,
    }


def one_hot_loss(
    _sender_input,
    _message,
    _receiver_input,
    receiver_output,
    labels,
    _aux_input,
):
    device = receiver_output.device
    accuracy = Mean(device=device)
    color_accuracy = Mean(device=device)
    shape_accuracy = Mean(device=device)
    size_accuracy = Mean(device=device)

    color_hits = _get_attribute_hits(receiver_output, labels, 0, 8)
    color_accuracy.update(color_hits)
    shape_hits = _get_attribute_hits(receiver_output, labels, 8, 11)
    shape_accuracy.update(shape_hits)
    size_hits = _get_attribute_hits(receiver_output, labels, 11, 12)
    size_accuracy.update(size_hits)

    complete_hits = color_hits * shape_hits * size_hits
    accuracy.update(complete_hits)

    loss = F.binary_cross_entropy(receiver_output, labels)

    logging.getLogger().setLevel(logging.ERROR)
    computed_accuracy = accuracy.compute()
    computed_color_accuracy = color_accuracy.compute()
    computed_shape_accuracy = shape_accuracy.compute()
    computed_size_accuracy = size_accuracy.compute()
    logging.getLogger().setLevel(logging.WARNING)

    return loss, {
        "accuracy": computed_accuracy.detach().clone().float(),
        "color_accuracy": computed_color_accuracy.detach().clone().float(),
        "shape_accuracy": computed_shape_accuracy.detach().clone().float(),
        "size_accuracy": computed_size_accuracy.detach().clone().float(),
    }


def _get_attribute_hits(output, ground_truth, start_index, end_index):
    ground_truth_attribute = ground_truth[:, start_index:end_index]
    output_attribute = F.one_hot(
        torch.argmax(output[:, start_index:end_index], dim=1),
        num_classes=ground_truth_attribute.shape[1],
    )
    hits = torch.sum(ground_truth_attribute * output_attribute, dim=1)

    return hits


def pixel_loss(
    _sender_input,
    _message,
    _receiver_input,
    receiver_output,
    labels,
    _aux_input,
):
    device = receiver_output.device
    accuracy = BinaryAccuracy(device=device)

    distances = torch.diagonal(torch.cdist(receiver_output, labels.float()))

    positives = torch.where(distances < 20, distances, 0)
    accuracy.update(positives, torch.ones_like(positives))

    return distances, {
        "accuracy": accuracy.compute().detach().clone().float(),
    }


def attention_loss(
    _sender_input,
    _message,
    _receiver_input,
    receiver_output,
    labels,
    _aux_input,
):
    device = receiver_output.device
    bb_probability_mass = Mean(device=device)

    loss = F.binary_cross_entropy(receiver_output, labels)
    bb_probability_mass.update(torch.sum(receiver_output * labels, dim=1))

    return loss, {
        "bb_probability_mass": bb_probability_mass.compute().detach().clone().float(),
    }
