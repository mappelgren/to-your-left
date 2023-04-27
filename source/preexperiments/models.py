from abc import ABC

import torch
from torch import nn
from torch.nn import Module
from torchvision.models import ResNet50_Weights, resnet50


class AbstractResnet(ABC, Module):
    def __init__(self, pretrained, fine_tune) -> None:
        super().__init__()

        if pretrained:
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            resnet = resnet50()

        # out 2048 * 7 * 7
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        if not fine_tune:
            for param in self.resnet.parameters():
                param.requires_grad = False
            self.resnet.eval()


class BoundingBoxClassifier(AbstractResnet):
    """
    Output:
     - classified bounding box (10 dimensions)

    Input:
     - bounding boxes of objects
    """

    def __init__(self, pretrained_resnet, fine_tune_resnet) -> None:
        super().__init__(pretrained_resnet, fine_tune_resnet)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(20480, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, data):
        data = data.permute(1, 0, 2, 3, 4)

        after_resnet = []
        for bounding_box in data:
            resnet = self.resnet(bounding_box)
            pooled = self.adaptive_pool(resnet)
            after_resnet.append(pooled)

        stacked = torch.stack(after_resnet)
        stacked = stacked.permute(1, 0, 2, 3, 4)

        classified = self.classifier(torch.flatten(stacked, start_dim=1))

        return classified


class CoordinatePredictor(AbstractResnet):
    """
    Output:
     - x and y coordinates of target object

    Input:
     - image
    """

    def __init__(self, pretrained_resnet, fine_tune_resnet) -> None:
        super().__init__(pretrained_resnet, fine_tune_resnet)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(100352, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
        )

    def forward(self, data):
        image, *_ = data

        resnet = self.resnet(image)
        pooled = self.adaptive_pool(resnet)
        classified = self.classifier(torch.flatten(pooled, start_dim=1))

        return classified


class AttributeCoordinatePredictor(AbstractResnet):
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
        pretrained_resnet,
        fine_tune_resnet,
    ) -> None:
        super().__init__(pretrained_resnet, fine_tune_resnet)
        # out 100_352
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.reduction = nn.Linear(100_352, 2048)

        self.dropout(nn.Dropout(0.5))

        self.predictor = nn.Linear(
            2048 + number_colors + number_shapes + number_sizes, 2
        )

    def forward(self, data):
        image, color_tensor, shape_tensor, size_tensor, *_ = data
        resnet = self.resnet(self.dropout(image))
        pooled = self.adaptive_pool(self.dropout(resnet))
        reduced = self.reduction(torch.flatten(pooled, start_dim=1))
        concatenated = torch.cat(
            (reduced, color_tensor, shape_tensor, size_tensor), dim=1
        )
        predicted = self.predictor(concatenated)

        return predicted


class AttributeLocationCoordinatePredictor(AbstractResnet):
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
        pretrained_resnet,
        fine_tune_resnet,
    ) -> None:
        super().__init__(pretrained_resnet, fine_tune_resnet)
        self.dropout = nn.Dropout(0.3)

        self.cnn = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # out 100_352
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.reduction = nn.Linear(100_352, 2048)

        self.predictor = nn.Linear(
            4608 + number_colors + number_shapes + number_sizes + (number_objects * 2),
            2,
        )

    def forward(self, data):
        image, color_tensor, shape_tensor, size_tensor, locations = data
        resnet = self.resnet(image)
        cnn = self.cnn(resnet)

        # pooled = self.adaptive_pool(self.dropout(resnet))
        # reduced = self.reduction(torch.flatten(pooled, start_dim=1))
        concatenated = torch.cat(
            (
                torch.flatten(cnn, start_dim=1),
                color_tensor,
                shape_tensor,
                size_tensor,
                locations,
            ),
            dim=1,
        )

        predicted = self.predictor(concatenated)

        return predicted


class MaskedCoordinatePredictor(AbstractResnet):
    """
    Output:
     - x and y coordinates of target object

    Input:
     - image
     - attributes (shape, size, color)
     - center coordinates of all objects
    """

    def __init__(self, pretrained_resnet, fine_tune_resnet) -> None:
        super().__init__(pretrained_resnet, fine_tune_resnet)
        self.dropout = nn.Dropout(0.3)

        self.cnn = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # out 100_352
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.reduction = nn.Linear(100_352, 2048)

        self.predictor = nn.Linear(
            4096,
            2,
        )

    def forward(self, data):
        image, _, _, _, _, masked_image, *_ = data
        resnet = self.resnet(image)
        pooled = self.adaptive_pool(self.dropout(resnet))
        reduced = self.reduction(torch.flatten(pooled, start_dim=1))

        masked_resnet = self.resnet(masked_image)
        masked_pooled = self.adaptive_pool(self.dropout(masked_resnet))
        masked_reduced = self.reduction(torch.flatten(masked_pooled, start_dim=1))

        concatenated = torch.cat(
            (reduced, masked_reduced),
            dim=1,
        )
        predicted = self.predictor(concatenated)

        return predicted


class ImageEncoder(AbstractResnet):
    def __init__(self, encoder_out_dim, pretrained_resnet, fine_tune_resnet) -> None:
        super().__init__(pretrained_resnet, fine_tune_resnet)
        # out 100_352
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.reduction = nn.Linear(100_352, encoder_out_dim)
        self.mean_reduction = nn.Linear(2048, encoder_out_dim)

    def forward(self, image):
        resnet = self.resnet(image)
        pooled = self.adaptive_pool(resnet)
        # reduced = self.reduction(torch.flatten(pooled, start_dim=1))
        flattened = torch.flatten(pooled, start_dim=2).permute(0, 2, 1)
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
