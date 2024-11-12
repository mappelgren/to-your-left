import torch
from torcheval.metrics import Mean
import torch.nn.functional as F

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
