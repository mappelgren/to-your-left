import torch
from mlt.preexperiments.models import CaptionDecoder
from mlt.shared_models import (
    BoundingBoxImageEncoder,
    CoordinateClassifier,
    ImageEncoder,
)
from torch import nn


class DummySender(nn.Module):
    def __init__(self, sender_hidden, *_args, **_kwargs) -> None:
        super().__init__()
        self.sender_hidden = sender_hidden

    def forward(self, x, _aux_input):
        return torch.rand((x.shape[0], self.sender_hidden), device=x.device)


class ReferentialGameSender(nn.Module):
    def __init__(
        self, sender_hidden, *_args, sender_image_embedding=256, **_kwargs
    ) -> None:
        super().__init__()
        self.image_encoder = BoundingBoxImageEncoder(
            image_embedding_dimension=sender_image_embedding
        )

        self.lin = nn.LazyLinear(sender_hidden, bias=False)

    def forward(self, x, _aux_input):
        encoded_images = []
        for image_index in range(x.shape[1]):
            encoded_images.append(self.image_encoder(x[:, image_index]))

        concatenated = torch.cat(encoded_images, dim=1)
        lin = self.lin(concatenated)

        return lin


class ReferentialGameReceiver(nn.Module):
    def __init__(self, *_args, receiver_image_embedding=256, **_kwargs) -> None:
        super().__init__()
        self.image_encoder = BoundingBoxImageEncoder(
            image_embedding_dimension=receiver_image_embedding
        )

        self.linear_message = nn.LazyLinear(receiver_image_embedding, bias=False)

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
        masked_image_encoder: ImageEncoder,
        sender_image_embedding: int,
        sender_hidden: int,
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
        receiver_image_embedding: int,
        caption_decoder: CaptionDecoder,
        encoded_sos,
        *_args,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.reduction = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(receiver_image_embedding)
        )
        self.caption_decoder = caption_decoder
        self.linear = nn.LazyLinear(caption_decoder.decoder_out)
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


class OneHotGeneratorReceiver(nn.Module):
    def __init__(
        self,
        image_encoder: ImageEncoder,
        receiver_projection: int,
        number_attributes: int,
        *_args,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.image_projection = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(receiver_projection)
        )

        self.message_projection = nn.LazyLinear(receiver_projection)

        self.attribute_predictor = nn.Sequential(
            nn.LazyLinear(number_attributes), nn.Softmax(dim=1)
        )

    def forward(self, message, x, _aux_input):
        image = x
        encoded_image = self.image_encoder(image).flatten(start_dim=2).permute(0, 2, 1)
        projected_image = self.image_projection(encoded_image)
        projected_message = self.message_projection(message)

        cat = torch.cat((projected_image, projected_message), dim=1)

        return self.attribute_predictor(cat)


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
        sender_encoder_vocab_size,
        sender_encoder_embedding,
        sender_encoder_out,
        image_encoder: ImageEncoder,
        sender_image_embedding,
        sender_hidden,
        *_args,
        **_kwargs,
    ) -> None:
        super().__init__()

        self.image_encoder = image_encoder
        self.reduction = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(sender_image_embedding)
        )

        self.embedding = nn.Embedding(
            sender_encoder_vocab_size, sender_encoder_embedding
        )
        self.lstm = nn.LSTM(
            sender_encoder_embedding, sender_encoder_out, batch_first=True
        )

        self.linear = nn.LazyLinear(sender_hidden)

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


class CoordinatePredictorReceiver(nn.Module):
    def __init__(
        self,
        image_encoder: ImageEncoder,
        receiver_image_embedding: int,
        coordinate_classifier: CoordinateClassifier,
        *_args,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.reduction = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(receiver_image_embedding)
        )

        self.coordinate_classifier = coordinate_classifier

    def forward(self, message, x, _aux_input):
        image = x
        processed_image = self.image_encoder(image)
        reduced = self.reduction(processed_image)

        concatenated = torch.cat((reduced, message), dim=1)
        coordinates = self.coordinate_classifier(concatenated)

        return coordinates


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


class DaleAttributeSender(nn.Module):
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
        sender_encoder_vocab_size,
        sender_encoder_embedding,
        sender_encoder_out,
        image_encoder: ImageEncoder,
        sender_projection,
        sender_hidden,
        *_args,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.image_projection = nn.LazyLinear(sender_projection)

        self.embedding = nn.Embedding(
            sender_encoder_vocab_size, sender_encoder_embedding
        )
        self.lstm = nn.LSTM(
            sender_encoder_embedding, sender_encoder_out, batch_first=True
        )
        self.attribute_projection = nn.LazyLinear(sender_projection)

        self.hidden = nn.LazyLinear(sender_hidden)

    def forward(self, x, aux_input):
        image = x
        attribute_tensor = aux_input["attribute_tensor"]

        encoded_image = self.image_encoder(image).flatten(start_dim=2).permute(0, 2, 1)
        projected_image = nn.functional.tanh(self.image_projection(encoded_image))

        embedded = self.embedding(attribute_tensor)
        _, (hidden_state, _) = self.lstm(embedded)
        projected_attributes = nn.functional.tanh(
            self.attribute_projection(hidden_state.squeeze())
        ).unsqueeze(2)

        dot = torch.matmul(projected_image, projected_attributes).squeeze()

        return self.hidden(dot)


class AttributeSender(nn.Module):
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
        sender_hidden,
        *_args,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.hidden = nn.LazyLinear(sender_hidden)

    def forward(self, _x, aux_input):
        attribute_tensor = aux_input["attribute_tensor"]

        return self.hidden(attribute_tensor)
