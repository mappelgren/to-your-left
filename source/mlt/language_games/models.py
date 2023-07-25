import torch
from mlt.preexperiments.data_readers import CaptionGeneratorDataset
from torch import nn


class ReferentialGameSender(nn.Module):
    def __init__(self, hidden_size, *_args, embedding_dimension=256, **_kwargs) -> None:
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(embedding_dimension, bias=False)
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
        self.image_encoder = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(embedding_dimension, bias=False)
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
        self, image_encoder, masked_image_encoder, hidden_size, *_args, **_kwargs
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.masked_image_encoder = masked_image_encoder

        self.linear = nn.LazyLinear(hidden_size)

    def forward(self, x, aux_input):
        image = x
        masked_image = aux_input["masked_image"]

        encoded_image = self.image_encoder(image)
        encoded_masked_image = self.masked_image_encoder(masked_image)

        concatenated = torch.cat((encoded_image, encoded_masked_image), dim=1)

        linear = self.linear(concatenated)
        return linear


class CaptionGeneratorReceiver(nn.Module):
    def __init__(
        self,
        image_encoder,
        masked_image_encoder,
        caption_decoder,
        encoded_sos,
        *_args,
        **_kwargs
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.masked_image_encoder = masked_image_encoder
        self.caption_decoder = caption_decoder
        self.linear = nn.LazyLinear(1024)
        self.encoded_sos = torch.tensor(encoded_sos)

    def forward(self, message, x, aux_input):
        image = x
        masked_image = aux_input["masked_image"]
        captions = aux_input["caption"]
        device = image.device

        if aux_input["train_mode"][0]:
            encoded_image = self.image_encoder(image)
            encoded_masked_image = self.masked_image_encoder(masked_image)

            concatenated = torch.cat(
                (encoded_image, encoded_masked_image, message), dim=1
            ).unsqueeze(dim=0)
            linear = self.linear(concatenated)
            lstm_states = linear, linear
            predicted, lstm_states = self.caption_decoder(captions[:, :-1], lstm_states)

            print(predicted)
            return predicted.permute(0, 2, 1)

        else:
            encoded_image = self.image_encoder(image)
            encoded_masked_image = self.masked_image_encoder(masked_image)

            concatenated = torch.cat(
                (encoded_image, encoded_masked_image, message), dim=1
            ).unsqueeze(dim=0)
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

            print(torch.stack(predicted).shape)

            return torch.stack(predicted).permute(1, 0, 2)
