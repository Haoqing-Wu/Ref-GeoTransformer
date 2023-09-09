import torch
import math
from torch.nn import Module, TransformerEncoder, TransformerEncoderLayer, Sequential, Linear, LayerNorm, ReLU, GELU, SiLU, ModuleList
from geotransformer.modules.transformer.vanilla_transformer import TransformerLayer

from timm.models.vision_transformer import Attention, Mlp
from einops import rearrange
from positional_encodings.torch_encodings import PositionalEncoding2D

class transformer(Module):
    # Build a transformer encoder
    def __init__(
            self, 
            n_layers=4,
            n_heads=4,
            query_dimensions=64,
            feed_forward_dimensions=2048,
            activation="gelu",
            time_emb_dim=256
        ):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = query_dimensions*n_heads
        self.half_dim = self.hidden_dim // 2
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
            #LayerNorm(query_dimensions*n_heads),
            Linear(query_dimensions*n_heads, 2)
            #ReLU(),
            #Linear(64, 32),
            #ReLU(),
            #Linear(32, 2)
        )
        self.feature_cross_attention = ModuleList([
            TransformerLayer(
                d_model=query_dimensions*n_heads, 
                num_heads=n_heads, 
                dropout=None, 
                activation_fn='ReLU'
            ) for _ in range(n_layers)
        ])
        self.feature_output_mlp = Sequential(
            LayerNorm(256),
            Linear(256, 512),
        )
        self.DiT_blocks = ModuleList([
            DiTBlock(query_dimensions*n_heads, n_heads, mlp_ratio=4.0) for _ in range(n_layers)
        ])
        self.feat0_proj = Sequential(
            Linear(self.half_dim, self.half_dim),
            ReLU(),
            Linear(self.half_dim, self.half_dim)
        )
        self.feat1_proj = Sequential(
            Linear(self.half_dim, self.half_dim),
            ReLU(),
            Linear(self.half_dim, self.half_dim)
        )
        self.time_emb = Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            Linear(time_emb_dim, n_heads*query_dimensions),
            GELU(),
            Linear(n_heads*query_dimensions, n_heads*query_dimensions)
        )
        self.feat_2d_mlp = Sequential(
            LayerNorm(768),
            Linear(768, 256)
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
    
    def mid_features_fusion(self, feats0, feats1):
        feats_matrix = []
        for i in range (feats0.shape[1]):
            feat0 = feats0[:, i, :, :]
            feat1 = feats1[:, i, :, :]
            feat_matrix = self.feature_fusion_dist(feat0, feat1)
            feats_matrix.append(feat_matrix)
        return feats_matrix
    
    def get_sel_indices_from_mask(self, mask):
        sel = torch.nonzero(mask.view(mask.shape[0], -1, 1), as_tuple=True)[1]
        sel_feat0_indices = torch.div(sel, mask.shape[2], rounding_mode='floor')
        sel_feat1_indices = sel % mask.shape[2]
        return sel_feat0_indices, sel_feat1_indices
    
    def feature_pooling_from_indices(self, feat, indices, pooling='mean'):
        feat = feat[:, indices, :]
        if pooling == 'mean':
            feat = torch.mean(feat, dim=1)
        if pooling == 'max':
            feat = torch.max(feat, dim=1)[0]
        return feat
        

    def forward(self, x_t, t, feats):

        feat0 = feats.get('ref_feats')
        feat1 = feats.get('src_feats')

        feat0_dist_emb = feats['ref_dist_emb']
        feat1_dist_emb = feats['src_dist_emb']
        dist_emb = self.feature_fusion_cat(feat0_dist_emb, feat1_dist_emb)
        dist_emb = torch.reshape(dist_emb, (dist_emb.shape[0], -1, dist_emb.shape[-1]))

        feat0_voxel_emb = feats['ref_voxel_emb']
        feat1_voxel_emb = feats['src_voxel_emb']
        voxel_emb = self.feature_fusion_add(feat0_voxel_emb, feat1_voxel_emb)
        voxel_emb = torch.reshape(voxel_emb, (voxel_emb.shape[0], -1, voxel_emb.shape[-1]))

        feat_2d = feats.get('feat_2d')
        c_2d = self.feat_2d_mlp(feat_2d)

        mid_feats0 = feats.get('ref_mid_feats')
        mid_feats1 = feats.get('src_mid_feats')
        mid_ctxs = self.mid_features_fusion(mid_feats0, mid_feats1)

        mask = feats.get('mask')
        sel_feat0_indices, sel_feat1_indices = self.get_sel_indices_from_mask(mask)
        feat0_global = self.feature_pooling_from_indices(feat0, sel_feat0_indices, pooling='mean')
        feat1_global = self.feature_pooling_from_indices(feat1, sel_feat1_indices, pooling='mean')
        #feat0 = torch.cat([feat0, feat0_global.unsqueeze(1).repeat(1, feat0.shape[1], 1)], dim=-1)
        #feat1 = torch.cat([feat1, feat1_global.unsqueeze(1).repeat(1, feat1.shape[1], 1)], dim=-1)

        #ctx = mid_ctxs[0]
        ctx = self.feature_fusion_cat(feat0, feat1)
        #ctx = mask.unsqueeze(-1) * ctx
        ctx = torch.reshape(ctx, (ctx.shape[0], -1, ctx.shape[-1]))

        

        x = x_t.squeeze(1).unsqueeze(-1)
        x = x.repeat(1, 1, 1, self.hidden_dim)
        x = torch.reshape(x, (x.shape[0], -1, x.shape[-1]))
        #x = x + dist_emb
        t = self.time_emb(t)
        #c = t + c_2d

        for i in range (self.n_layers):
            x = x + ctx
            #x, _ = self.feature_cross_attention[i](x, ctx)
            x = self.DiT_blocks[i](x, t)
        x = self.output_mlp(x)
        #x = x[:, :-1, :]
        x = torch.reshape(x, (x_t.shape[0], x_t.shape[2], x_t.shape[3], -1))
        x = rearrange(x, 'b h w c -> b c h w')
        return x






def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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