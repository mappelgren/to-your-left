from torch import nn
import torch

from shared_models import ImageEncoder


class DummySender(nn.Module):
    def __init__(self, sender_hidden, *_args, **_kwargs) -> None:
        super().__init__()
        self.sender_hidden = sender_hidden

    def forward(self, x, _aux_input):
        return torch.rand((x.shape[0], self.sender_hidden), device=x.device)



class MaskedCoordinatePredictorSender(nn.Module):
    """
    Output:
     - x and y coordinates of target object

    Input:
     - image
     - attributes (shape, size, color)
     - center coordinates of all objects
    """

    def __init__(
        self,
        image_encoder: ImageEncoder,
        masked_image_encoder: ImageEncoder,
        sender_image_embedding: int,
        sender_hidden,
        *_args,
        **_kwargs,
    ) -> None:
        super().__init__()

        self.image_encoder = image_encoder
        self.masked_image_encoder = masked_image_encoder
        self.reduction = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(sender_image_embedding)
        )

        self.linear = nn.LazyLinear(sender_hidden)

    def forward(self, x, aux_input):
        image = x
        masked_image = aux_input["masked_image"]

        reduced = self.image_encoder(image)
        reduced_masked_image = self.masked_image_encoder(masked_image)

        concatenated = torch.cat(
            (reduced, reduced_masked_image),
            dim=1,
        )
        reduced = self.reduction(concatenated)

        hidden = self.linear(reduced)

        return hidden

class AttentionPredictorReceiver(nn.Module):
    def __init__(
            self, image_encoder: ImageEncoder, receiver_projection: int, *_args, **_kwargs
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.image_projection = nn.LazyLinear(receiver_projection)

        self.message_projection = nn.LazyLinear(receiver_projection)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, message, x, _aux_input):
        image = x
        encoded_image = self.image_encoder(image).flatten(start_dim=2).permute(0, 2, 1)
        projected_image = nn.functional.tanh(self.image_projection(encoded_image))

        projected_message = nn.functional.tanh(
            self.message_projection(message.squeeze())
        ).unsqueeze(2)

        dot = torch.matmul(projected_image, projected_message).squeeze()

        return self.softmax(dot)



class CaptionDecoder(nn.Module):
    def __init__(
        self, decoder_vocab_size, decoder_embedding, decoder_out, *_args, **_kwargs
    ) -> None:
        super().__init__()
        self.decoder_out = decoder_out
        self.embeddings = nn.Embedding(decoder_vocab_size, decoder_embedding)
        self.lstm = nn.LSTM(decoder_embedding, decoder_out, batch_first=True)
        self.classifier = nn.Linear(decoder_out, decoder_vocab_size)

    def forward(self, caption, input_states):
        embedded = self.embeddings(caption)
        output, output_states = self.lstm(embedded, input_states)
        prediction = self.classifier(output)
        return prediction, output_states

