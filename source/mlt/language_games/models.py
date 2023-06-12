import torch
from torch import nn


class ReferentialGameSender(nn.Module):
    def __init__(self, hidden_size, embedding_dimension=256) -> None:
        super().__init__()
        self.image_encoder = nn.LazyLinear(embedding_dimension, bias=False)

        self.lin = nn.LazyLinear(hidden_size, bias=False)

    def forward(self, x, _aux_input):
        encoded_images = []
        for image_index in range(x.shape[1]):
            encoded_images.append(self.image_encoder(x[:, image_index]))

        concatenated = torch.cat(encoded_images, dim=1)
        lin = self.lin(concatenated)

        return lin


class ReferentialGameReceiver(nn.Module):
    def __init__(self, embedding_dimension=256) -> None:
        super().__init__()
        self.image_encoder = nn.LazyLinear(embedding_dimension, bias=False)

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


class MaskedDaleSender(nn.Module):
    def __init__(self, image_encoder, masked_image_encoder) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.masked_image_encoder = masked_image_encoder

    def forward(self, x, _aux_input):
        image, masked_image = x

        encoded_image = self.image_encoder(image).unsqueeze(dim=0)
        encoded_masked_image = self.masked_image_encoder(masked_image).unsqueeze(dim=0)
        concatenated = torch.cat((encoded_image, encoded_masked_image), dim=2)

        return concatenated


class DaleReceiver(nn.Module):
    def __init__(self, image_encoder, caption_decoder, encoded_sos) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.caption_decoder = caption_decoder
        self.encoded_sos = torch.tensor(encoded_sos)

    def forward(self, x, _input, _aux_input):
        image, caption = _input

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
