import random

import torch
from mlt.shared_models import (
    BoundingBoxImageEncoder,
    CoordinateClassifier,
    ImageEncoder,
    MaskedImageEncoder,
)
from torch import nn


class BoundingBoxClassifier(nn.Module):
    """
    Output:
     - classified bounding box (10 dimensions)

    Input:
     - bounding boxes of objects
    """

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()

        self.classifier_2 = nn.Sequential(nn.Flatten(), nn.LazyLinear(2))
        self.classifier_5 = nn.Sequential(nn.Flatten(), nn.LazyLinear(5))
        self.classifier_10 = nn.Sequential(nn.Flatten(), nn.LazyLinear(10))

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, data):
        bounding_boxes, *_ = data

        if bounding_boxes.shape[1] == 2:
            output = self.classifier_2(bounding_boxes)
        elif bounding_boxes.shape[1] == 5:
            output = self.classifier_5(bounding_boxes)
        elif bounding_boxes.shape[1] == 10:
            output = self.classifier_10(bounding_boxes)

        return self.softmax(output)


class BoundingBoxCaptionGenerator(nn.Module):
    def __init__(
        self,
        decoder_out_dim,
        caption_decoder,
        encoded_sos,
        *_args,
        image_embedding_dimension=256,
        **_kwargs
    ) -> None:
        super().__init__()
        self.image_encoder = BoundingBoxImageEncoder(
            image_embedding_dimension=image_embedding_dimension
        )

        self.lin = nn.LazyLinear(decoder_out_dim, bias=False)
        self.caption_decoder = caption_decoder
        self.encoded_sos = torch.tensor(encoded_sos)

    def forward(self, data):
        bounding_boxes, caption, *_ = data

        encoded_images = []
        for image_index in range(bounding_boxes.shape[1]):
            encoded_images.append(self.image_encoder(bounding_boxes[:, image_index]))

        concatenated = torch.cat(encoded_images, dim=1).unsqueeze(dim=0)

        lin = self.lin(concatenated)
        lstm_states = lin, lin
        predicted, lstm_states = self.caption_decoder(caption[:, :-1], lstm_states)

        return predicted.permute(0, 2, 1)

    def caption(self, bounding_boxes):
        device = bounding_boxes.device

        encoded_images = []
        for image_index in range(bounding_boxes.shape[1]):
            encoded_images.append(self.image_encoder(bounding_boxes[:, image_index]))

        concatenated = torch.cat(encoded_images, dim=1).unsqueeze(dim=0)
        lin = self.lin(concatenated)
        lstm_states = lin, lin

        caption = []
        # shape: batch, sequence length
        word = torch.full((bounding_boxes.shape[0], 1), self.encoded_sos, device=device)
        for _ in range(3):
            predicted_word_layer, lstm_states = self.caption_decoder(word, lstm_states)
            word = torch.max(predicted_word_layer, dim=2).indices
            caption.append(word)

        return torch.cat(caption, dim=1)


class BoundingBoxAttributeClassifier(nn.Module):
    """
    Output:
     - classified bounding box (10 dimensions)

    Input:
     - bounding boxes of objects
    """

    def __init__(self, image_embedding_dimension, *_args, **_kwargs) -> None:
        super().__init__()

        # self.image_encoder = image_encoder
        self.reduction = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(image_embedding_dimension)
        )

        self.linear_attributes = nn.LazyLinear(image_embedding_dimension)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, data):
        bounding_boxes, attribute_tensor, *_ = data

        attribute_tensor = self.linear_attributes(attribute_tensor)
        dot_products = []
        for image_index in range(bounding_boxes.shape[1]):
            reduced = self.reduction(bounding_boxes[:, image_index])
            dot_products.append(torch.sum(attribute_tensor * reduced, dim=1))

        output = torch.stack(dot_products, dim=1)

        return self.softmax(output)


class CoordinatePredictor(nn.Module):
    """
    Output:
     - x and y coordinates of target object

    Input:
     - image
    """

    def __init__(
        self,
        image_encoder: ImageEncoder,
        image_embedding_dimension: int,
        coordinate_classifier: CoordinateClassifier,
        *_args,
        **_kwargs
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.reduction = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(image_embedding_dimension)
        )
        self.coordinate_classifier = coordinate_classifier

    def forward(self, data):
        image, *_ = data

        encoded_image = self.image_encoder(image)
        reduced = self.reduction(encoded_image)
        coordinates = self.coordinate_classifier(reduced)

        return coordinates


class AttributeCoordinatePredictor(nn.Module):
    """
    Output:
     - x and y coordinates of target object

    Input:
     - image
     - attributes (shape, size, color)
    """

    def __init__(
        self,
        image_encoder: ImageEncoder,
        image_embedding_dimension: int,
        coordinate_classifier: CoordinateClassifier,
        *_args,
        **_kwargs
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.reduction = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(image_embedding_dimension)
        )

        self.coordinate_classifier = coordinate_classifier

    def forward(self, data):
        image, attribute_tensor, *_ = data

        encoded_image = self.image_encoder(image)
        reduced = self.reduction(encoded_image)
        concatenated = torch.cat((reduced, attribute_tensor), dim=1)
        coordinates = self.coordinate_classifier(concatenated)

        return coordinates


class AttributeLocationCoordinatePredictor(nn.Module):
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
        image_embedding_dimension: int,
        coordinate_classifier: CoordinateClassifier,
        *_args,
        **_kwargs
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.reduction = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(image_embedding_dimension)
        )

        self.coordinate_classifier = coordinate_classifier

    def forward(self, data):
        image, attribute_tensor, locations, *_ = data

        encoded_image = self.image_encoder(image)
        reduced = self.reduction(encoded_image)

        concatenated = torch.cat(
            (
                reduced,
                attribute_tensor,
                locations,
            ),
            dim=1,
        )

        coordinates = self.coordinate_classifier(concatenated)

        return coordinates


class DaleAttributeCoordinatePredictor(nn.Module):
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
        image_embedding_dimension,
        embedding_dim,
        encoder_out_dim,
        image_encoder: ImageEncoder,
        coordinate_classifier: CoordinateClassifier,
        *_args,
        **_kwargs
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.reduction = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(image_embedding_dimension)
        )

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, encoder_out_dim, batch_first=True)

        self.coordinate_classifier = coordinate_classifier

    def forward(self, data):
        image, attribute_tensor, *_ = data

        encoded_image = self.image_encoder(image)
        reduced = self.reduction(encoded_image)

        embedded = self.embedding(attribute_tensor)
        _, (hidden_state, _) = self.lstm(embedded)

        concatenated = torch.cat((reduced, hidden_state.squeeze()), dim=1)
        coordinates = self.coordinate_classifier(concatenated)

        return coordinates


class MaskedDaleAttributeCoordinatePredictor(nn.Module):
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
        image_embedding_dimension,
        embedding_dim,
        encoder_out_dim,
        image_encoder: ImageEncoder,
        masked_image_encoder: MaskedImageEncoder,
        coordinate_classifier: CoordinateClassifier,
        *_args,
        **_kwargs
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.masked_image_encoder = masked_image_encoder
        self.reduction = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(image_embedding_dimension)
        )

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, encoder_out_dim, batch_first=True)

        self.coordinate_classifier = coordinate_classifier

    def forward(self, data):
        image, attribute_tensor, _, masked_image, *_ = data

        encoded_image = self.image_encoder(image)
        encoded_masked_image = self.masked_image_encoder(masked_image)
        concatenated = torch.cat(
            (encoded_image, encoded_masked_image),
            dim=1,
        )

        reduced = self.reduction(concatenated)

        embedded = self.embedding(attribute_tensor)
        _, (hidden_state, _) = self.lstm(embedded)

        concatenated = torch.cat((reduced, hidden_state.squeeze()), dim=1)
        coordinates = self.coordinate_classifier(concatenated)

        return coordinates


class MaskedCoordinatePredictor(nn.Module):
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
        image_embedding_dimension: int,
        coordinate_classifier: CoordinateClassifier,
        *_args,
        **_kwargs
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.masked_image_encoder = masked_image_encoder
        self.reduction = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(image_embedding_dimension)
        )

        self.coordinate_classifier = coordinate_classifier

    def forward(self, data):
        image, _, _, masked_image, *_ = data

        encoded_image = self.image_encoder(image)
        encoded_masked_image = self.masked_image_encoder(masked_image)

        concatenated = torch.cat(
            (encoded_image, encoded_masked_image),
            dim=1,
        )
        reduced = self.reduction(concatenated)
        coordinates = self.coordinate_classifier(reduced)

        return coordinates


class RandomCoordinatePredictor(nn.Module):
    """
    Output:
     - x and y coordinates of target object

    Input:
    """

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()

        self.dummy = nn.Linear(1, 1)

    def forward(self, data):
        image, *_ = data
        # 224 = image width and height
        return (
            torch.rand((image.shape[0], 2), device=image.device, requires_grad=True)
            * 224
        )


class CaptionDecoder(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, decoder_out_dim, *_args, **_kwargs
    ) -> None:
        super().__init__()
        self.decoder_out_dim = decoder_out_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, decoder_out_dim, batch_first=True)
        self.classifier = nn.Linear(decoder_out_dim, vocab_size)

    def forward(self, caption, input_states):
        embedded = self.embeddings(caption)
        output, output_states = self.lstm(embedded, input_states)
        prediction = self.classifier(output)
        return prediction, output_states


class CaptionGenerator(nn.Module):
    """
    Output:
     - caption

    Input:
     - image
    """

    def __init__(
        self,
        image_encoder: ImageEncoder,
        image_embedding_dimension: int,
        caption_decoder,
        encoded_sos,
        *_args,
        **_kwargs
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.reduction = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(image_embedding_dimension)
        )
        self.caption_decoder = caption_decoder
        self.encoded_sos = torch.tensor(encoded_sos)

    def forward(self, data):
        image, caption, *_ = data

        encoded_image = self.image_encoder(image)
        reduced = self.reduction(encoded_image).unsqueeze(dim=0)
        lstm_states = reduced, reduced
        predicted, lstm_states = self.caption_decoder(caption[:, :-1], lstm_states)

        return predicted.permute(0, 2, 1)

    def caption(self, image):
        device = image.device

        encoded_image = self.image_encoder(image)
        reduced = self.reduction(encoded_image).unsqueeze(dim=0)

        caption = []
        lstm_states = reduced, reduced

        # shape: batch, sequence length
        word = torch.full((image.shape[0], 1), self.encoded_sos, device=device)
        for _ in range(3):
            predicted_word_layer, lstm_states = self.caption_decoder(word, lstm_states)
            word = torch.max(predicted_word_layer, dim=2).indices
            caption.append(word)

        return torch.cat(caption, dim=1)


class MaskedCaptionGenerator(nn.Module):
    """
    Output:
     - caption

    Input:
     - image
    """

    def __init__(
        self,
        image_encoder: ImageEncoder,
        masked_image_encoder: MaskedImageEncoder,
        image_embedding_dimension: int,
        caption_decoder,
        encoded_sos,
        *_args,
        **_kwargs
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.masked_image_encoder = masked_image_encoder
        self.caption_decoder = caption_decoder
        self.encoded_sos = torch.tensor(encoded_sos)
        self.reduction = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(image_embedding_dimension)
        )

    def forward(self, data):
        image, caption, _, masked_image, *_ = data

        encoded_image = self.image_encoder(image)
        encoded_masked_image = self.masked_image_encoder(masked_image)
        concatenated = torch.cat((encoded_image, encoded_masked_image), dim=1)
        reduced = self.reduction(concatenated).unsqueeze(dim=0)

        lstm_states = reduced, reduced
        predicted, lstm_states = self.caption_decoder(caption[:, :-1], lstm_states)

        return predicted.permute(0, 2, 1)

    def caption(self, image, masked_image):
        device = image.device

        encoded_image = self.image_encoder(image)
        encoded_masked_image = self.masked_image_encoder(masked_image)
        concatenated = torch.cat((encoded_image, encoded_masked_image), dim=1)
        reduced = self.reduction(concatenated).unsqueeze(dim=0)

        caption = []
        lstm_states = reduced, reduced

        # shape: batch, sequence length
        word = torch.full((image.shape[0], 1), self.encoded_sos, device=device)
        for _ in range(3):
            predicted_word_layer, lstm_states = self.caption_decoder(word, lstm_states)
            word = torch.max(predicted_word_layer, dim=2).indices
            caption.append(word)

        return torch.cat(caption, dim=1)
