import torch
import math
from torch.nn import Module, TransformerEncoder, TransformerEncoderLayer, Sequential, Linear, LayerNorm, ReLU, GELU, SiLU, ModuleList
from geotransformer.modules.transformer.vanilla_transformer import TransformerLayer
from timm.models.vision_transformer import Attention, Mlp
from einops import rearrange
from positional_encodings.torch_encodings import PositionalEncoding1D

class transformer(Module):
    # Build a transformer encoder
    def __init__(
            self, 
            n_layers=4,
            n_heads=4,
            query_dimensions=64,
            feed_forward_dimensions=2048,
            activation="gelu",
            time_emb_dim=256,
            pos_emb_dim=256
        ):
        super().__init__()

        self.output_mlp = Sequential(
            Linear(query_dimensions*n_heads, 2)
        )
        self.feature_cross_attension = TransformerLayer(
            d_model=256, num_heads=8, dropout=None, activation_fn='ReLU'
        )
        self.feature_output_mlp = Sequential(
            LayerNorm(256),
            Linear(256, 512),
        )
        self.DiT_blocks = ModuleList([
            DiTBlock(query_dimensions*n_heads, n_heads, mlp_ratio=4.0) for _ in range(n_layers)
        ])
        self.time_emb = Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            Linear(time_emb_dim, n_heads*query_dimensions),
            GELU(),
            Linear(n_heads*query_dimensions, n_heads*query_dimensions)
        )
        self.pos_emb = PositionalEncoding1D(pos_emb_dim)
        self.feat_2d_mlp = Sequential(
            LayerNorm(768),
            Linear(768, 256)
        )
        self.feat_3d_mlp = Sequential(
            LayerNorm(512),
            Linear(512, 256)
        )

    def feature_fusion_cat(self, feat0, feat1):
        feat_matrix = torch.cat([feat0.unsqueeze(2).repeat(1, 1, feat1.shape[1], 1),
                                    feat1.unsqueeze(1).repeat(1, feat0.shape[1], 1, 1)], dim=-1)
        feat_matrix = feat_matrix.view(feat0.shape[0], feat0.shape[1], feat1.shape[1], -1)
        return feat_matrix

    def feature_fusion_add(self, feat0, feat1):
        feat_matrix = feat0.unsqueeze(2).repeat(1, 1, feat1.shape[1], 1) + feat1.unsqueeze(1).repeat(1, feat0.shape[1], 1, 1)
        feat_matrix = feat_matrix.view(feat0.shape[0], feat0.shape[1], feat1.shape[1], -1)
        return feat_matrix
    
    def feature_fusion_dist(self, feat0, feat1):
        feat_matrix = torch.abs(feat0.unsqueeze(2).repeat(1, 1, feat1.shape[1], 1) - feat1.unsqueeze(1).repeat(1, feat0.shape[1], 1, 1))
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

        
    def forward(self, x_t, t, feats):

        feat_2d = feats.get('feat_2d')
        feat_3d = feats.get('feat_3d')
        x = x_t.squeeze(1)
        x = x.unsqueeze(-1).repeat(1, 1, 256)
        x = x + self.pos_emb(x)

        t = self.time_emb(t)
        c_2d = self.feat_2d_mlp(feat_2d)
        c_3d = self.feat_3d_mlp(feat_3d)
        #c = c_2d + c_3d
        c = c_3d
        c = t + c

        for block in self.DiT_blocks:
            x = block(x, c)
        x = self.output_mlp(x)

        x = rearrange(x, 'b h c -> b c h')
        return x






def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    #return shift.unsqueeze(1) * torch.cos(x * torch.pi) + scale.unsqueeze(1) * torch.sin(x * torch.pi)


class DiTBlock(Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = GELU
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = Sequential(
            SiLU(),
            Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(x, shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class SinusoidalPositionEmbeddings(Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


if __name__ == "__main__":
    # Test the transformer encoder
    x = torch.rand(10, 512, 64*4).cuda()
    transformer = transformer().cuda()
    y = transformer(x)
    print(y.shape)