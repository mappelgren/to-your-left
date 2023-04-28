import math
from abc import ABC, abstractmethod

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, VGG16_Weights, resnet50, vgg16


class FeatureExtractor(ABC, nn.Module):
    @property
    @abstractmethod
    def feature_shape(self):
        ...


class ResnetFeatureExtractor(FeatureExtractor):
    def __init__(self, pretrained, fine_tune) -> None:
        super().__init__()

        if pretrained:
            self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.resnet = resnet50()

        if not fine_tune:
            for param in self.resnet.parameters():
                param.requires_grad = False
            self.resnet.eval()

        # self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        # self.resnet.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.resnet.fc = nn.Identity()

        self._feature_shape = (2048, 1, 1)

    @property
    def feature_shape(self):
        return self._feature_shape

    def forward(self, data):
        return self.resnet(data)


class VggFeatureExtractor(FeatureExtractor):
    def __init__(self, pretrained, fine_tune) -> None:
        super().__init__()

        if pretrained:
            self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        else:
            self.vgg = vgg16()

        classifier = nn.Sequential(list(self.vgg.children())[-1][:-3])
        self.vgg.classifier = classifier
        self._feature_shape = (4096,)

        if not fine_tune:
            for param in self.vgg.parameters():
                param.requires_grad = False
            self.vgg.eval()

    @property
    def feature_shape(self):
        return self._feature_shape

    def forward(self, data):
        return self.vgg(data)


class BoundingBoxClassifier(nn.Module):
    """
    Output:
     - classified bounding box (10 dimensions)

    Input:
     - bounding boxes of objects
    """

    def __init__(
        self, feature_extractor: FeatureExtractor, inputs_are_features=False
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.input_are_feature = inputs_are_features
        classifier_in_dim = math.prod(feature_extractor.feature_shape)

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, data):
        data = data.permute(1, 0, 2, 3, 4)

        if self.input_are_feature:
            stacked = data
        else:
            after_feature_extractor = []
            for bounding_box in data:
                after_feature_extractor.append(self.feature_extractor(bounding_box))
            stacked = torch.stack(after_feature_extractor)
            stacked = stacked.permute(1, 0, 2, 3, 4)

        classified = self.classifier(torch.flatten(stacked, start_dim=1))

        return classified


class CoordinatePredictor(nn.Module):
    """
    Output:
     - x and y coordinates of target object

    Input:
     - image
    """

    def __init__(
        self, feature_extractor: FeatureExtractor, inputs_are_features=False
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.inputs_are_features = inputs_are_features

        classifier_in_dim = math.prod(feature_extractor.feature_shape)

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
        )

    def forward(self, data):
        image, *_ = data

        if self.inputs_are_features:
            extracted_features = image
        else:
            extracted_features = self.feature_extractor(image)

        classified = self.classifier(torch.flatten(extracted_features, start_dim=1))

        return classified


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
        number_colors,
        number_shapes,
        number_sizes,
        feature_extractor: FeatureExtractor,
        inputs_are_features=False,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.inputs_are_features = inputs_are_features
        classifier_in_dim = math.prod(feature_extractor.feature_shape)

        self.reduction = nn.Linear(classifier_in_dim, 2048)

        self.dropout = nn.Dropout(0.2)

        self.predictor = nn.Linear(
            2048 + number_colors + number_shapes + number_sizes, 2
        )

    def forward(self, data):
        image, attribute_tensor, *_ = data

        if self.inputs_are_features:
            extracted_features = image
        else:
            extracted_features = self.feature_extractor(image)

        reduced = self.dropout(
            self.reduction(torch.flatten(extracted_features, start_dim=1))
        )
        concatenated = torch.cat((reduced, attribute_tensor), dim=1)
        predicted = self.predictor(concatenated)

        return predicted


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
        number_colors,
        number_shapes,
        number_sizes,
        number_objects,
        feature_extractor: FeatureExtractor,
        inputs_are_features=False,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.inputs_are_features = inputs_are_features
        classifier_in_dim = math.prod(feature_extractor.feature_shape)

        self.dropout = nn.Dropout(0.3)

        self.cnn = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.reduction = nn.Linear(classifier_in_dim, 2048)

        self.predictor = nn.Linear(
            4608 + number_colors + number_shapes + number_sizes + (number_objects * 2),
            2,
        )

    def forward(self, data):
        image, attribute_tensor, locations = data

        if self.inputs_are_features:
            extracted_features = image
        else:
            extracted_features = self.feature_extractor(image)

        cnn = self.cnn(extracted_features)

        # reduced = self.reduction(torch.flatten(pooled, start_dim=1))
        concatenated = torch.cat(
            (
                torch.flatten(cnn, start_dim=1),
                attribute_tensor,
                locations,
            ),
            dim=1,
        )

        predicted = self.predictor(concatenated)

        return predicted


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
        embedding_dim,
        encoder_out_dim,
        feature_extractor: FeatureExtractor,
        inputs_are_features=False,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.2)

        self.feature_extractor = feature_extractor
        self.inputs_are_features = inputs_are_features
        classifier_in_dim = math.prod(feature_extractor.feature_shape)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, encoder_out_dim, batch_first=True)

        self.reduction = nn.Linear(classifier_in_dim, 1024)
        self.relu = nn.ReLU()

        self.predictor = nn.Linear(
            1024 + encoder_out_dim,
            2,
        )

    def forward(self, data):
        image, attribute_tensor, *_ = data

        if self.inputs_are_features:
            extracted_features = image
        else:
            extracted_features = self.feature_extractor(image)

        reduced = self.dropout(
            self.reduction(torch.flatten(extracted_features, start_dim=1))
        )

        embedded = self.embedding(attribute_tensor)
        _, (hidden_state, _) = self.lstm(embedded)

        concatenated = self.relu(torch.cat((reduced, hidden_state.squeeze()), dim=1))
        predicted = self.predictor(concatenated)

        return predicted


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
        feature_extractor: FeatureExtractor,
        inputs_are_features=False,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.3)

        self.feature_extractor = feature_extractor
        self.inputs_are_features = inputs_are_features
        classifier_in_dim = math.prod(feature_extractor.feature_shape)

        self.cnn = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.reduction = nn.Linear(classifier_in_dim, 2048)

        self.predictor = nn.Linear(
            4096,
            2,
        )

    def forward(self, data):
        image, _, _, masked_image, *_ = data

        if self.inputs_are_features:
            extracted_features = image
        else:
            extracted_features = self.feature_extractor(image)

        extracted_features = self.dropout(extracted_features)
        reduced = self.reduction(torch.flatten(extracted_features, start_dim=1))

        if self.inputs_are_features:
            masked_extracted_features = self.feature_extractor(masked_image)
        else:
            masked_extracted_features = masked_image

        masked_extracted_features = self.dropout(masked_extracted_features)
        masked_reduced = self.reduction(
            torch.flatten(masked_extracted_features, start_dim=1)
        )

        concatenated = torch.cat(
            (reduced, masked_reduced),
            dim=1,
        )
        predicted = self.predictor(concatenated)

        return predicted


class ImageEncoder(nn.Module):
    def __init__(
        self,
        encoder_out_dim,
        feature_extractor: FeatureExtractor,
        inputs_are_features=False,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.inputs_are_features = inputs_are_features
        classifier_in_dim = math.prod(feature_extractor.feature_shape)

        self.reduction = nn.Linear(classifier_in_dim, encoder_out_dim)
        self.mean_reduction = nn.Linear(2048, encoder_out_dim)

    def forward(self, image):
        if self.inputs_are_features:
            extracted_features = image
        else:
            extracted_features = self.feature_extractor(image)
        # reduced = self.reduction(torch.flatten(pooled, start_dim=1))
        flattened = torch.flatten(extracted_features, start_dim=2).permute(0, 2, 1)
        reduced = self.mean_reduction(flattened.mean(dim=1))
        return reduced


class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, decoder_out_dim) -> None:
        super().__init__()
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

    def __init__(self, image_encoder, caption_decoder, encoded_sos) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.caption_decoder = caption_decoder
        self.encoded_sos = torch.tensor(encoded_sos)

    def forward(self, data):
        image, caption, *_ = data

        encoded_image = self.image_encoder(image).unsqueeze(dim=0)
        lstm_states = encoded_image, encoded_image
        predicted, lstm_states = self.caption_decoder(caption[:, :-1], lstm_states)

        return predicted.permute(0, 2, 1)

    def caption(self, image):
        device = image.device

        encoded_image = self.image_encoder(image).unsqueeze(dim=0)

        caption = []
        lstm_states = encoded_image, encoded_image

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

    def __init__(self, image_encoder, caption_decoder, encoded_sos) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.caption_decoder = caption_decoder
        self.encoded_sos = torch.tensor(encoded_sos)

    def forward(self, data):
        image, caption, _, masked_image, *_ = data

        encoded_image = self.image_encoder(image).unsqueeze(dim=0)
        encoded_masked_image = self.image_encoder(masked_image).unsqueeze(dim=0)
        concatenated = torch.cat((encoded_image, encoded_masked_image), dim=2)

        lstm_states = concatenated, concatenated
        predicted, lstm_states = self.caption_decoder(caption[:, :-1], lstm_states)

        return predicted.permute(0, 2, 1)

    def caption(self, image, masked_image):
        device = image.device

        encoded_image = self.image_encoder(image).unsqueeze(dim=0)
        encoded_masked_image = self.image_encoder(masked_image).unsqueeze(dim=0)
        concatenated = torch.cat((encoded_image, encoded_masked_image), dim=2)

        caption = []
        lstm_states = concatenated, concatenated

        # shape: batch, sequence length
        word = torch.full((image.shape[0], 1), self.encoded_sos, device=device)
        for _ in range(3):
            predicted_word_layer, lstm_states = self.caption_decoder(word, lstm_states)
            word = torch.max(predicted_word_layer, dim=2).indices
            caption.append(word)

        return torch.cat(caption, dim=1)
