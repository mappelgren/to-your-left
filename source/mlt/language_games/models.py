import torch
from mlt.shared_models import (
    BoundingBoxImageEncoder,
    CoordinateClassifier,
    ImageEncoder,
    MaskedImageEncoder,
)
from torch import nn


class DummyReferentialSender(nn.Module):
    def __init__(self, hidden_size, *_args, **_kwargs) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x, _aux_input):
        return torch.rand((x.shape[0], self.hidden_size), device=x.device)


class ReferentialGameSender(nn.Module):
    def __init__(self, hidden_size, *_args, embedding_dimension=256, **_kwargs) -> None:
        super().__init__()
        self.image_encoder = BoundingBoxImageEncoder(
            embedding_dimension=embedding_dimension
        )

        self.lin = nn.LazyLinear(hidden_size, bias=False)

    def forward(self, x, _aux_input):
        encoded_images = []
        for image_index in range(x.shape[1]):
            encoded_images.append(self.image_encoder(x[:, image_index]))

        concatenated = torch.cat(encoded_images, dim=1)
        lin = self.lin(concatenated)

        return lin


class ReferentialGameReceiver(nn.Module):
    def __init__(self, *_args, embedding_dimension=256, **_kwargs) -> None:
        super().__init__()
        self.image_encoder = BoundingBoxImageEncoder(
            embedding_dimension=embedding_dimension
        )

        self.linear_message = nn.LazyLinear(embedding_dimension, bias=False)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, message, x, _aux_input):
        message = self.linear_message(message)
        dot_products = []
        for image_index in range(x.shape[1]):
            encoded_image = self.image_encoder(x[:, image_index])
            dot_products.append(torch.sum(message * encoded_image, dim=1))

        output = torch.stack(dot_products, dim=1)

        return self.softmax(output)


class CaptionGeneratorSender(nn.Module):
    def __init__(
        self,
        image_encoder: ImageEncoder,
        masked_image_encoder: MaskedImageEncoder,
        embedding_dimension: int,
        hidden_size,
        *_args,
        **_kwargs
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.masked_image_encoder = masked_image_encoder
        self.reduction = nn.Sequential(nn.Flatten(), nn.LazyLinear(embedding_dimension))

        self.linear = nn.LazyLinear(hidden_size)

    def forward(self, x, aux_input):
        image = x
        masked_image = aux_input["masked_image"]

        encoded_image = self.image_encoder(image)
        encoded_masked_image = self.masked_image_encoder(masked_image)

        concatenated = torch.cat((encoded_image, encoded_masked_image), dim=1)

        reduced = self.reduction(concatenated)

        linear = self.linear(reduced)
        return linear


class CaptionGeneratorReceiver(nn.Module):
    def __init__(
        self,
        image_encoder: ImageEncoder,
        embedding_dimension: int,
        caption_decoder,
        encoded_sos,
        *_args,
        **_kwargs
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.reduction = nn.Sequential(nn.Flatten(), nn.LazyLinear(embedding_dimension))
        self.caption_decoder = caption_decoder
        self.linear = nn.LazyLinear(1024)
        self.encoded_sos = torch.tensor(encoded_sos)

    def forward(self, message, x, aux_input):
        image = x
        captions = aux_input["caption"]
        device = image.device

        if aux_input["train_mode"][0]:
            encoded_image = self.image_encoder(image)
            reduced = self.reduction(encoded_image)

            concatenated = torch.cat((reduced, message), dim=1).unsqueeze(dim=0)
            linear = self.linear(concatenated)
            lstm_states = linear, linear
            predicted, lstm_states = self.caption_decoder(captions[:, :-1], lstm_states)

            return predicted.permute(0, 2, 1)

        else:
            encoded_image = self.image_encoder(image)
            reduced = self.reduction(encoded_image)

            concatenated = torch.cat((reduced, message), dim=1).unsqueeze(dim=0)
            linear = self.linear(concatenated)
            lstm_states = linear, linear

            predicted = []
            # shape: batch, sequence length
            word = torch.full((image.shape[0], 1), self.encoded_sos, device=device)
            for _ in range(3):
                predicted_word_layer, lstm_states = self.caption_decoder(
                    word, lstm_states
                )
                word = torch.max(predicted_word_layer, dim=2).indices
                predicted.append(predicted_word_layer.squeeze(dim=1))

            return torch.stack(predicted).permute(1, 2, 0)


class DaleAttributeCoordinatePredictorSender(nn.Module):
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
        vocab_size,
        embedding_dimension,
        encoder_out_dim,
        image_encoder: ImageEncoder,
        image_embedding_dimension,
        hidden_size,
        *_args,
        **_kwargs
    ) -> None:
        super().__init__()

        self.image_encoder = image_encoder
        self.reduction = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(image_embedding_dimension)
        )

        self.embedding = nn.Embedding(vocab_size, embedding_dimension)
        self.lstm = nn.LSTM(embedding_dimension, encoder_out_dim, batch_first=True)

        self.linear = nn.LazyLinear(hidden_size)

    def forward(self, x, aux_input):
        image = x
        attribute_tensor = aux_input["attribute_tensor"]

        processed_image = self.image_encoder(image)
        reduced = self.reduction(processed_image)

        embedded = self.embedding(attribute_tensor)
        _, (hidden_state, _) = self.lstm(embedded)

        concatenated = torch.cat((reduced, hidden_state.squeeze()), dim=1)

        hidden = self.linear(concatenated)

        return hidden


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
        masked_image_encoder: MaskedImageEncoder,
        embedding_dimension: int,
        hidden_size,
        *_args,
        **_kwargs
    ) -> None:
        super().__init__()

        self.image_encoder = image_encoder
        self.masked_image_encoder = masked_image_encoder
        self.reduction = nn.Sequential(nn.Flatten(), nn.LazyLinear(embedding_dimension))

        self.linear = nn.LazyLinear(hidden_size)

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


class CoordinatePredictorReceiver(nn.Module):
    def __init__(
        self,
        image_encoder: ImageEncoder,
        embedding_dimension: int,
        coordinate_classifier: CoordinateClassifier,
        *_args,
        **_kwargs
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.reduction = nn.Sequential(nn.Flatten(), nn.LazyLinear(embedding_dimension))

        self.coordinate_classifier = coordinate_classifier

    def forward(self, message, x, _aux_input):
        image = x
        processed_image = self.image_encoder(image)
        reduced = self.reduction(processed_image)

        concatenated = torch.cat((reduced, message), dim=1)
        coordinates = self.coordinate_classifier(concatenated)

        return coordinates
