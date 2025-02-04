import torch
from torch.nn import Module, TransformerEncoder, TransformerEncoderLayer, Sequential, Linear, LayerNorm, ReLU


class transformer(Module):
    # Build a transformer encoder
    def __init__(
            self, 
            n_layers=4,
            n_heads=4,
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
        self.output_mlp = Sequential(
            LayerNorm(query_dimensions*n_heads),
            Linear(query_dimensions*n_heads, 64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, 1)
        )

        
    def forward(self, x_t, t, ctx):
        t_seq = t.unsqueeze(1)

        x = x_t.unsqueeze(-1) + ctx
        x = torch.reshape(x, (x.shape[0], -1, x.shape[-1]))
        x = torch.cat([x, t_seq], dim=1)
        x = self.transformer_encoder(x)
        x = self.output_mlp(x)
        x = x[:, :-1, :]
        x = torch.reshape(x, (x.shape[0], -1, x_t.shape[-1]))
        return x
if __name__ == "__main__":
    # Test the transformer encoder
    x = torch.rand(10, 512, 64*4).cuda()
    transformer = transformer().cuda()
    y = transformer(x)
    print(y.shape)