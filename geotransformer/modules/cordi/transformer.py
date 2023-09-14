import torch
import torch.nn.functional as F
import math
from torch.nn import Module, TransformerEncoder, TransformerEncoderLayer, Sequential, Linear, LayerNorm, ReLU, GELU, SiLU, ModuleList
from geotransformer.modules.transformer.vanilla_transformer import TransformerLayer

from timm.models.vision_transformer import Attention, Mlp
from einops import rearrange
from positional_encodings.torch_encodings import PositionalEncoding2D
from geotransformer.modules.transformer.rpe_transformer import RPEMultiHeadAttention
from geotransformer.modules.layers import build_dropout_layer, build_act_layer

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
        self.output_mlp = Finallayer(query_dimensions*n_heads, 2)
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
        # for new implementation
        self.n_ref = 38
        self.n_src = 80
        self.d_model = query_dimensions*n_heads
        self.input_proj_ref = Linear(self.n_src, self.d_model)
        self.input_proj_src = Linear(self.n_ref, self.d_model)
        self.transformer = RPEDiT(
            hidden_dim=self.d_model,
            num_heads=n_heads,
            blocks=['self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross'],
            n_ref=self.n_ref,
            n_src=self.n_src
        )
        #self.final_layer = Finallayer(self.d_model, 2)
        self.final_layer = Sequential(
            LayerNorm(self.d_model),
            Linear(self.d_model, 256),
            ReLU(),
            Linear(256, 1)
        )
        self.split = Sequential(
            Linear(1, 256),
            ReLU(),
            Linear(256, 2)
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
        

    def _forward(self, x_t, t, feats):

        feat0 = feats.get('ref_feats')
        feat1 = feats.get('src_feats')
        #feat0 = feats.get('ref_knn_feats')
        #feat1 = feats.get('src_knn_feats')

        feat0_dist_emb = feats['ref_geo_emb']
        feat1_dist_emb = feats['src_geo_emb']
        dist_emb = self.feature_fusion_add(feat0_dist_emb, feat1_dist_emb)
        dist_emb = torch.reshape(dist_emb, (dist_emb.shape[0], -1, dist_emb.shape[-1]))

        feat0_voxel_emb = feats['ref_voxel_emb']
        feat1_voxel_emb = feats['src_voxel_emb']
        voxel_emb = self.feature_fusion_add(feat0_voxel_emb, feat1_voxel_emb)
        voxel_emb = torch.reshape(voxel_emb, (voxel_emb.shape[0], -1, voxel_emb.shape[-1]))

        feat0_knn_emb = feats['ref_knn_emb']
        feat1_knn_emb = feats['src_knn_emb']
        knn_emb = self.feature_fusion_cat(feat0_knn_emb, feat1_knn_emb)
        knn_emb = torch.reshape(knn_emb, (knn_emb.shape[0], -1, knn_emb.shape[-1]))

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
        #x = x + dist_emb + ctx
        t = self.time_emb(t)
        #c = t + c_2d
        x = x + ctx
        for i in range (self.n_layers):
            
            #x, _ = self.feature_cross_attention[i](x, ctx)
            x = self.DiT_blocks[i](x, t)
        x = self.output_mlp(x, t)
        #x = x[:, :-1, :]
        x = torch.reshape(x, (x_t.shape[0], x_t.shape[2], x_t.shape[3], -1))
        x = rearrange(x, 'b h w c -> b c h w')
        return x

    def forward(self, x_t, t, feats):
        x = x_t.squeeze(1)
        ref_x = x
        src_x = x.transpose(1, 2)

        ref_geo_emb = feats['ref_geo_emb']
        src_geo_emb = feats['src_geo_emb']
        ref_voxel_emb = feats['ref_voxel_emb']
        src_voxel_emb = feats['src_voxel_emb']

        t_emb = self.time_emb(t)

        ref_x, src_x = self.transformer(
            ref_x,
            src_x,
            ref_geo_emb,
            src_geo_emb,
            ref_voxel_emb,
            src_voxel_emb,
            t_emb,
        )
        #src_x = src_x.transpose(1, 2)
        #x = torch.mean(torch.stack((ref_x, src_x)), dim=0)
        #x = self.split(x)
        x = ref_x.unsqueeze(2).repeat(1, 1, src_x.shape[1], 1) + src_x.unsqueeze(1).repeat(1, ref_x.shape[1], 1, 1)
        x = self.final_layer(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def _check_block_type(block):
    if block not in ['self', 'cross']:
        raise ValueError('Unsupported block type "{}".'.format(block))

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
        

    def forward(self, x1, x2, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        adaLN = {}
        adaLN['shift_msa'] = shift_msa
        adaLN['scale_msa'] = scale_msa
        adaLN['gate_msa'] = gate_msa
        adaLN['shift_mlp'] = shift_mlp
        adaLN['scale_mlp'] = scale_mlp
        adaLN['gate_mlp'] = gate_mlp

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(x, shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    
        

    
class RPEAttentionLayer(Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(RPEAttentionLayer, self).__init__()
        self.attention = RPEMultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.linear = Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = LayerNorm(d_model)

    def forward(
        self,
        input_states,
        memory_states,
        position_states,
        adaLN,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ):
        shift_msa = adaLN['shift_msa']
        scale_msa = adaLN['scale_msa']
        gate_msa = adaLN['gate_msa']
        mod_states = modulate(self.norm(input_states), shift_msa, scale_msa)
        #mod_states = modulate(input_states, shift_msa, scale_msa)
        hidden_states, attention_scores = self.attention(
            mod_states,
            memory_states,
            memory_states,
            position_states,
            key_weights=memory_weights,
            key_masks=memory_masks,
            attention_factors=attention_factors,
        )

        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = input_states + gate_msa.unsqueeze(1) * hidden_states
        return output_states, attention_scores

class AttentionOutput(Module):
    def __init__(self, d_model, dropout=None, activation_fn='ReLU'):
        super(AttentionOutput, self).__init__()
        self.expand = Linear(d_model, d_model * 2)
        self.activation = build_act_layer(activation_fn)
        self.squeeze = Linear(d_model * 2, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = LayerNorm(d_model)

    def forward(self, input_states, adaLN):
        shift_mlp = adaLN['shift_mlp']
        scale_mlp = adaLN['scale_mlp']
        gate_mlp = adaLN['gate_mlp']

        mod_states = modulate(self.norm(input_states), shift_mlp, scale_mlp)
        hidden_states = self.expand(mod_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.squeeze(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = input_states + gate_mlp.unsqueeze(1) * hidden_states
        return output_states
    
class RPETransformerLayer(Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='ReLU'):
        super(RPETransformerLayer, self).__init__()
        self.attention = RPEAttentionLayer(d_model, num_heads, dropout=dropout)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)

    def forward(
        self,
        input_states,
        memory_states,
        position_states,
        adaLN,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            position_states,
            adaLN,
            memory_weights=memory_weights,
            memory_masks=memory_masks,
            attention_factors=attention_factors,
        )
        output_states = self.output(hidden_states, adaLN)
        return output_states, attention_scores


class RPEConditionalTransformer(Module):
    def __init__(
        self,
        blocks,
        d_model,
        num_heads,
        dropout=None,
        activation_fn='ReLU',
        return_attention_scores=False,
        parallel=False,
    ):
        super(RPEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers0 = []
        layers1 = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers0.append(RPETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
                layers1.append(RPETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
            else:
                layers0.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
                layers1.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers0 = ModuleList(layers0)
        self.layers1 = ModuleList(layers1)
        self.return_attention_scores = return_attention_scores
        self.parallel = parallel


    def forward(self, feats0, feats1, embeddings0, embeddings1, adaLN, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers0[i](feats0, feats0, embeddings0, adaLN, memory_masks=masks0)
                feats1, scores1 = self.layers1[i](feats1, feats1, embeddings1, adaLN, memory_masks=masks1)
            else:
                if self.parallel:
                    new_feats0, scores0 = self.layers0[i](feats0, feats1, memory_masks=masks1)
                    new_feats1, scores1 = self.layers1[i](feats1, feats0, memory_masks=masks0)
                    feats0 = new_feats0
                    feats1 = new_feats1
                else:
                    feats0, scores0 = self.layers0[i](feats0, feats1, memory_masks=masks1)
                    feats1, scores1 = self.layers1[i](feats1, feats0, memory_masks=masks0)

            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1
        
class RPEDiT(Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        blocks,
        n_ref,
        n_src,
        dropout=None,
        activation_fn='ReLU',
    ):
        super(RPEDiT, self).__init__()
        self.n_ref = n_ref
        self.n_src = n_src
        self.input_proj0 = Linear(n_src, hidden_dim)
        self.input_proj1 = Linear(n_ref, hidden_dim)
        self.transformer = RPEConditionalTransformer(
            blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn
        )
        self.final_layer0 = Finallayer(hidden_dim, n_src)
        self.final_layer1 = Finallayer(hidden_dim, n_ref)
        self.adaLN_modulation = Sequential(
            SiLU(),
            Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

    def forward(
            self,
            feats0_in,
            feats1_in,
            geo_emb0,
            geo_emb1,
            voxel_emb0,
            voxel_emb1,
            t_emb,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=1)
        adaLN = {}
        adaLN['shift_msa'] = shift_msa
        adaLN['scale_msa'] = scale_msa
        adaLN['gate_msa'] = gate_msa
        adaLN['shift_mlp'] = shift_mlp
        adaLN['scale_mlp'] = scale_mlp
        adaLN['gate_mlp'] = gate_mlp

        feats0 = self.input_proj0(feats0_in)
        feats1 = self.input_proj1(feats1_in)
        feats0 = feats0 + voxel_emb0
        feats1 = feats1 + voxel_emb1

        feats0, feats1 = self.transformer(
            feats0,
            feats1,
            geo_emb0,
            geo_emb1,
            adaLN
        )
        feats0 = F.normalize(feats0, p=2, dim=1)
        feats1 = F.normalize(feats1, p=2, dim=1)
        #feats0 = self.final_layer0(feats0, t_emb).view(feats0.shape[0], -1, self.n_src, 1)
        #feats1 = self.final_layer1(feats1, t_emb).view(feats1.shape[0], -1, self.n_ref, 1)
        #feats0 = feats0 + feats0_in.unsqueeze(-1)
        #feats1 = feats1 + feats1_in.unsqueeze(-1)

        return feats0, feats1

    
class Finallayer(Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = Sequential(
            SiLU(),
            Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
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