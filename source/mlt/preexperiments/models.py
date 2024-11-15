import torch
from mlt.shared_models import (
    BoundingBoxImageEncoder,
    CoordinateClassifier,
    ImageEncoder,
    MaskPredictor,
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
        decoder_out,
        caption_decoder,
        encoded_sos,
        *_args,
        image_embedding=256,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.image_encoder = BoundingBoxImageEncoder(
            image_embedding_dimension=image_embedding
        )

        self.lin = nn.LazyLinear(decoder_out, bias=False)
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

    def __init__(self, image_embedding, *_args, **_kwargs) -> None:
        super().__init__()

        # self.image_encoder = image_encoder
        self.reduction = nn.Sequential(nn.Flatten(), nn.LazyLinear(image_embedding))

        self.linear_attributes = nn.LazyLinear(image_embedding)

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
        image_embedding: int,
        coordinate_classifier: CoordinateClassifier,
        *_args,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.reduction = nn.Sequential(nn.Flatten(), nn.LazyLinear(image_embedding))
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
        image_embedding: int,
        coordinate_classifier: CoordinateClassifier,
        *_args,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.reduction = nn.Sequential(nn.Flatten(), nn.LazyLinear(image_embedding))

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
        image_embedding: int,
        coordinate_classifier: CoordinateClassifier,
        *_args,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.reduction = nn.Sequential(nn.Flatten(), nn.LazyLinear(image_embedding))

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
        encoder_vocab_size,
        encoder_embedding,
        encoder_out,
        image_embedding,
        image_encoder: ImageEncoder,
        coordinate_classifier: CoordinateClassifier,
        *_args,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.reduction = nn.Sequential(nn.Flatten(), nn.LazyLinear(image_embedding))

        self.embedding = nn.Embedding(encoder_vocab_size, encoder_embedding)
        self.lstm = nn.LSTM(encoder_embedding, encoder_out, batch_first=True)

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
        encoder_vocab_size,
        encoder_embedding,
        encoder_out,
        image_embedding,
        image_encoder: ImageEncoder,
        masked_image_encoder: ImageEncoder,
        coordinate_classifier: CoordinateClassifier,
        *_args,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.masked_image_encoder = masked_image_encoder
        self.reduction = nn.Sequential(nn.Flatten(), nn.LazyLinear(image_embedding))

        self.embedding = nn.Embedding(encoder_vocab_size, encoder_embedding)
        self.lstm = nn.LSTM(encoder_embedding, encoder_out, batch_first=True)

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
        masked_image_encoder: ImageEncoder,
        image_embedding: int,
        coordinate_classifier: CoordinateClassifier,
        *_args,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.masked_image_encoder = masked_image_encoder
        self.reduction = nn.Sequential(nn.Flatten(), nn.LazyLinear(image_embedding))

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


class MaskedMaskPredictor(nn.Module):
    """
    Output:
     - mask around the target object

    Input:
     - image
     - masked image
    """

    def __init__(
        self,
        image_encoder: ImageEncoder,
        masked_image_encoder: ImageEncoder,
        image_embedding: int,
        mask_predictor: MaskPredictor,
        *_args,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.masked_image_encoder = masked_image_encoder
        self.reduction = nn.Sequential(nn.Flatten(), nn.LazyLinear(image_embedding))

        self.mask_predictor = mask_predictor

    def forward(self, data):
        image, _, _, masked_image, *_ = data

        encoded_image = self.image_encoder(image)
        encoded_masked_image = self.masked_image_encoder(masked_image)

        concatenated = torch.cat(
            (encoded_image, encoded_masked_image),
            dim=1,
        )
        reduced = self.reduction(concatenated)
        mask = self.mask_predictor(reduced)

        return mask


class DaleAttributeAttentionPredictor(nn.Module):
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
        projection,
        encoder_vocab_size,
        encoder_embedding,
        encoder_out,
        image_encoder: ImageEncoder,
        *_args,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.image_projection = nn.LazyLinear(projection)

        self.embedding = nn.Embedding(encoder_vocab_size, encoder_embedding)
        self.lstm = nn.LSTM(encoder_embedding, encoder_out, batch_first=True)
        self.attribute_projection = nn.LazyLinear(projection)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        image, attribute_tensor, *_ = data

        encoded_image = self.image_encoder(image).flatten(start_dim=2).permute(0, 2, 1)
        projected_image = nn.functional.tanh(self.image_projection(encoded_image))

        embedded = self.embedding(attribute_tensor)
        _, (hidden_state, _) = self.lstm(embedded)
        projected_attributes = nn.functional.tanh(
            self.attribute_projection(hidden_state.squeeze())
        ).unsqueeze(2)

        dot = torch.matmul(projected_image, projected_attributes).squeeze()

        return self.softmax(dot)


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
        image_embedding: int,
        caption_decoder,
        encoded_sos,
        *_args,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.reduction = nn.Sequential(nn.Flatten(), nn.LazyLinear(image_embedding))
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
        masked_image_encoder: ImageEncoder,
        image_embedding: int,
        caption_decoder,
        encoded_sos,
        *_args,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.masked_image_encoder = masked_image_encoder
        self.caption_decoder = caption_decoder
        self.encoded_sos = torch.tensor(encoded_sos)
        self.reduction = nn.Sequential(nn.Flatten(), nn.LazyLinear(image_embedding))

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


class OneHotGenerator(nn.Module):
    """
    Output:
     - caption

    Input:
     - image
    """

    def __init__(
        self,
        image_encoder: ImageEncoder,
        encoder_vocab_size: int,
        encoder_embedding: int,
        encoder_out: int,
        projection: int,
        number_attributes: int,
        *_args,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.image_projection = nn.Sequential(nn.Flatten(), nn.LazyLinear(projection))

        self.embedding = nn.Embedding(encoder_vocab_size, encoder_embedding)
        self.lstm = nn.LSTM(encoder_embedding, encoder_out, batch_first=True)
        self.attribute_projection = nn.LazyLinear(projection)

        self.attribute_predictor = nn.Sequential(
            nn.LazyLinear(number_attributes), nn.Softmax(dim=1)
        )

    def forward(self, data):
        image, attribute_tensor, *_ = data

        encoded_image = self.image_encoder(image).flatten(start_dim=2).permute(0, 2, 1)
        projected_image = self.image_projection(encoded_image)

        embedded = self.embedding(attribute_tensor)
        _, (hidden_state, _) = self.lstm(embedded)
        projected_attributes = self.attribute_projection(hidden_state.squeeze())

        cat = torch.cat((projected_image, projected_attributes), dim=1)

        return self.attribute_predictor(cat)
