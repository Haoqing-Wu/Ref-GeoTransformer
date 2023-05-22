import torch
from torch.nn import Module, TransformerEncoder, TransformerEncoderLayer


class transformer(Module):
    # Build a transformer encoder
    def __init__(
            self, 
            n_layers=8,
            n_heads=8,
            query_dimensions=64,
            feed_forward_dimensions=2048,
            activation="gelu"
        ):
        super().__init__()
        self.encoder_layer = TransformerEncoderLayer(
            d_model=query_dimensions*n_heads,
            nhead=n_heads,
            dim_feedforward=feed_forward_dimensions,
            dropout=0.1,
            activation=activation,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=n_layers
        )
        
    def forward(self, x):
        return self.transformer_encoder(x)

if __name__ == "__main__":
    # Test the transformer encoder
    x = torch.rand(10, 512, 64*8).cuda()
    transformer = transformer().cuda()
    y = transformer(x)
    print(y.shape)