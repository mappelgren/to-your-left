import torch
import torch.nn.functional as F
from torcheval.metrics import BinaryAccuracy, Mean, MulticlassAccuracy


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
    non_target_accuracy = BinaryAccuracy(device=device)

    predicted_captions = receiver_output.max(dim=1).indices
    for sample, output_sample in zip(
        aux_input["non_target_captions"], predicted_captions
    ):
        for caption in sample:
            described_non_target_object = torch.tensor(False)
            if torch.equal(output_sample, caption):
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

    loss = F.cross_entropy(receiver_output, labels)

    return loss, {
        "accuracy": accuracy.compute().detach().clone().float(),
        "word-by-word_accuracy": word_by_word_accuracy.compute()
        .detach()
        .clone()
        .float(),
        "non_target_accuracy": non_target_accuracy.compute().detach().clone().float(),
    }


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
    mean = Mean(device=device)

    distances = torch.diagonal(torch.cdist(receiver_output, labels.float()))
    mean.update(distances)

    positives = torch.where(distances < 20, distances, 0)
    accuracy.update(positives, torch.ones_like(positives))

    return mean.compute(), {
        "accuracy": accuracy.compute().detach().clone().float(),
    }
