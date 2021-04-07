from torch import nn

class SimpleEncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        latent = self.encoder(x)
        pred = self.decoder(latent)
        assert pred.shape[2:] == x.shape[2:], f"shape is wrong after inference, shapes possibly not compatible with model arch\n\
                {x.shape} -> {pred.shape}"
        return {"prediction": pred}
