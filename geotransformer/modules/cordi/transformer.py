import torch
from torch.nn import Module, TransformerEncoder, TransformerEncoderLayer, Sequential, Linear, LayerNorm, ReLU
from geotransformer.modules.transformer.vanilla_transformer import TransformerLayer

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
        self.feature_cross_attension = TransformerLayer(
            d_model=256, num_heads=8, dropout=None, activation_fn='ReLU'
        )
        self.feature_output_mlp = Sequential(
            LayerNorm(256),
            Linear(256, 512),
        )

    def feature_fusion_cat(self, feat0, feat1):
        feat_matrix = torch.cat([feat0.unsqueeze(2).repeat(1, 1, feat1.shape[1], 1),
                                    feat1.unsqueeze(1).repeat(1, feat0.shape[1], 1, 1)], dim=-1)
        feat_matrix = feat_matrix.view(feat0.shape[0], feat0.shape[1], feat1.shape[1], -1)
        return feat_matrix
    
    def feature_fusion_cross_attention(self, feat0, feat1):
        feat0 = feat0.squeeze(0)
        feat1 = feat1.squeeze(0)
        feat0 = feat0.unsqueeze(1).repeat(1, feat1.shape[0], 1)
        feat1 = feat1.unsqueeze(0).repeat(feat0.shape[0], 1, 1)
        feat_matrix0, _ = self.feature_cross_attension(feat0, feat1)
        feat_matrix1, _ = self.feature_cross_attension(feat1, feat0)
        feat_matrix = torch.cat([feat_matrix0, feat_matrix1], dim=-1)
        feat_matrix = feat_matrix.unsqueeze(0)
        return feat_matrix

        
    def forward(self, x_t, t, feat0, feat1):
        
        ctx = self.feature_fusion_cat(feat0, feat1)
        #ctx = self.feature_fusion_cross_attention(feat0, feat1)
        x = x_t.unsqueeze(-1) + ctx
        x = torch.reshape(x, (x.shape[0], -1, x.shape[-1]))
        t_seq = t.unsqueeze(1)
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