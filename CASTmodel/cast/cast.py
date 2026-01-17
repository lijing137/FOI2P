import torch
import torch.nn as nn
from pytorch3d.ops import knn_points, knn_gather

from CASTmodel.utils import pairwise_distance

from CASTmodel.cast.spot_attention import SparseTransformerLayer, Upsampling, Downsampling

from CASTmodel.utils import pairwise_distance
from CASTmodel.transformerCAST.pe_transformer import PETransformerLayer
from CASTmodel.transformerCAST.rpe_transformer import RPETransformerLayer
from CASTmodel.transformerCAST.vanilla_transformer import TransformerLayer
from CASTmodel.transformerCAST.positional_encoding import GeometricStructureEmbedding
from CASTmodel.transformerCAST.linear_transformer import LinearTransformerLayer


class SpotGuidedGeoTransformer(nn.Module):
    def __init__(self, cfg):
        super(SpotGuidedGeoTransformer, self).__init__()
        self.k = cfg.k            # num of neighbor points whose corresponding patches are candidate spots
        self.spots = cfg.spots    # num of neighbor points whose corresponding patches are selected as spots
        self.down_k = cfg.down_k  # num of nodes for down sampling and fusion with semi-dense points
        self.spot_k = cfg.spot_k  # num of points in a spot
        self.blocks = cfg.blocks
        self.sigma_c = cfg.sigma_c
        self.seed_num = cfg.seed_num
        self.seed_threshold = cfg.seed_threshold
        self.dual_normalization = cfg.dual_normalization

        self.in_proj1 = nn.Linear(cfg.input_dim_c, cfg.hidden_dim)
        self.in_proj2 = nn.Linear(cfg.input_dim_f, cfg.hidden_dim)
        self.linear_cross_attention = LinearTransformerLayer(
            cfg.hidden_dim, cfg.num_heads, cfg.dropout, cfg.activation_fn
        )
        if "sigma_d" in cfg.keys() and "sigma_a" in cfg.keys():
            self.embed = GeometricStructureEmbedding(
                cfg.hidden_dim, cfg.sigma_d, cfg.sigma_a, cfg.angle_k, cfg.reduction_a
            )
            self.geometric_structure_embedding = True
        else: self.geometric_structure_embedding = False

        self.upsampling = nn.ModuleList()
        self.downsampling = nn.ModuleList()
        self.self_attentions = nn.ModuleList()
        self.cross_attentions = nn.ModuleList()
        self.spot_guided_attentions = nn.ModuleList()
        self.consistency_aware_attentions = nn.ModuleList()

        for _ in range(self.blocks):
            self.upsampling.append(Upsampling(cfg.hidden_dim, cfg.hidden_dim))
            self.downsampling.append(Downsampling(cfg.hidden_dim, cfg.hidden_dim))
            if self.geometric_structure_embedding:
                self.self_attentions.append(RPETransformerLayer(
                    cfg.hidden_dim, cfg.num_heads, cfg.dropout, cfg.activation_fn
                ))
            else:
                self.self_attentions.append(PETransformerLayer(
                    cfg.hidden_dim, cfg.num_heads, cfg.dropout, cfg.activation_fn
                ))
            self.cross_attentions.append(TransformerLayer(
                cfg.hidden_dim, cfg.num_heads, cfg.dropout, cfg.activation_fn
            ))
            self.spot_guided_attentions.append(SparseTransformerLayer(
                cfg.hidden_dim, cfg.num_heads, False, cfg.dropout, cfg.activation_fn
            ))
            self.consistency_aware_attentions.append(SparseTransformerLayer(
                cfg.hidden_dim, cfg.num_heads, True, cfg.dropout, cfg.activation_fn
            ))
    
    def matching_scores(self, input_states:torch.Tensor, memory_states:torch.Tensor):
        if input_states.ndim == 2:
            matching_scores = torch.einsum('mc,nc->mn', input_states, memory_states)
        else:
            matching_scores = torch.einsum('bmc,bnc->bmn', input_states, memory_states)
        if self.dual_normalization:
            ref_matching_scores = torch.softmax(matching_scores, dim=-1)
            src_matching_scores = torch.softmax(matching_scores, dim=-2)
            matching_scores = ref_matching_scores * src_matching_scores
        return matching_scores
    
    @torch.no_grad()
    def compatibility_scores(self, ref_dists, src_dists, matching_indices):
        """
        Args:
            ref_dists (Tensor): (B, N, N)
            src_dists (Tensor): (B, M, M)
            matching_indices (Tensor): (B, N, 1)

        Returns:
            compatibility (Tensor): (B, N, N)
        """
        src_dists = knn_gather(src_dists, matching_indices).squeeze(2)  # (B, N, 1, M)
        src_dists = knn_gather(src_dists.transpose(1, 2), matching_indices).squeeze(2)  # (B, N, N)
        return torch.relu(1. - torch.abs(ref_dists - src_dists) / self.sigma_c)  # (B, N, N)
    
    @torch.no_grad()
    def seeding(self, compatible_scores:torch.Tensor, confidence_scores:torch.Tensor):
        selection_scores = compatible_scores.lt(compatible_scores.max(-1, True)[0] * self.seed_threshold)  # (B, N)
        max_num = torch.clamp_max(selection_scores.gt(0).int().sum(-1).min(), self.seed_num)
        selection_scores = selection_scores.float() * confidence_scores.squeeze(-1)  # (B, N)
        return selection_scores.topk(max_num, dim=-1).indices  # (B, K)
    
    def forward(self,ref_points,src_points, ref_feats,src_feats, ref_points_c,src_points_c, ref_feats_c,src_feats_c):
        """
        Args:
            ref_points (Tensor): (B, N, 3)
            src_points (Tensor): (B, M, 3)
            ref_feats (Tensor): (B, N, C)
            src_feats (Tensor): (B, M, C)
            ref_points_c (Tensor): (B, N', 3)
            src_points_c (Tensor): (B, M', 3)
            ref_feats_c (Tensor): (B, N', C')
            src_feats_c (Tensor): (B, M', C')

        Returns:
            ref_feats: torch.Tensor (B, N, C)
            src_feats: torch.Tensor (B, M, C)
            matching_scores: List[torch.Tensor] (B, N, M)
        """
        k = max(self.k + 1, self.spot_k)
        with torch.no_grad():
            ref_dists = pairwise_distance(ref_points, ref_points)  # (B, N, N)
            src_dists = pairwise_distance(src_points, src_points)  # (B, M, M)
            ref_idx = ref_dists.topk(k, largest=False).indices  # (B, N, k)
            src_idx = src_dists.topk(k, largest=False).indices  # (B, M, k)

            # for nearest up-sampling fusion
            ref_idx_up = knn_points(ref_points, ref_points_c)[1]  # (B, N, 1)
            src_idx_up = knn_points(src_points, src_points_c)[1]  # (B, M, 1)
        
        # for knn interpolation in down-sampling fusion
        _, ref_idx_down, ref_xyz_down = knn_points(ref_points_c, ref_points, K=self.down_k, return_nn=True)
        _, src_idx_down, src_xyz_down = knn_points(src_points_c, src_points, K=self.down_k, return_nn=True)
        
        ref_feats_c = self.in_proj1(ref_feats_c)
        src_feats_c = self.in_proj1(src_feats_c)
        
        ref_feats = self.in_proj2(ref_feats)
        src_feats = self.in_proj2(src_feats)
        
        if self.geometric_structure_embedding:
            ref_embeddings = self.embed(ref_points_c)
            src_embeddings = self.embed(src_points_c)

        # 对1/4特征进行线性交叉注意力融合
        new_ref_feats = self.linear_cross_attention(ref_feats, src_feats)
        new_src_feats = self.linear_cross_attention(src_feats, ref_feats)

        correlation = []
        ref_compatibility = []
        src_compatibility = []

        for i in range(self.blocks):
            # 使用1/8特征进行自注意力融合
            if self.geometric_structure_embedding:
                ref_feats_c,_ = self.self_attentions[i](ref_feats_c, ref_feats_c, ref_embeddings)
                src_feats_c,_ = self.self_attentions[i](src_feats_c, src_feats_c, src_embeddings)
            else:
                ref_feats_c,_ = self.self_attentions[i](ref_feats_c, ref_feats_c, ref_points_c, ref_points_c)
                src_feats_c,_ = self.self_attentions[i](src_feats_c, src_feats_c, src_points_c, src_points_c)
            # 接着上面进行自注意力进行交叉注意力融合
            new_ref_feats_c,_ = self.cross_attentions[i](ref_feats_c, src_feats_c)
            new_src_feats_c,_ = self.cross_attentions[i](src_feats_c, ref_feats_c)

            # 进行上、下采样进行1/4和1/8的融合
            ref_feats = self.upsampling[i](new_ref_feats, new_ref_feats_c, ref_idx_up)
            src_feats = self.upsampling[i](new_src_feats, new_src_feats_c, src_idx_up)
            
            ref_feats_c = self.downsampling[i](new_ref_feats_c, new_ref_feats, ref_points_c, ref_xyz_down, ref_idx_down)
            src_feats_c = self.downsampling[i](new_src_feats_c, new_src_feats, src_points_c, src_xyz_down, src_idx_down)

            # 计算相似度矩阵
            matching_scores = self.matching_scores(ref_feats, src_feats)
            correlation.append(matching_scores)
            # 计算匹配矩阵的最高得分和索引
            confidence_scores, matching_indices = torch.max(matching_scores, dim=-1, keepdim=True)
            compatible_scores = self.compatibility_scores(ref_dists, src_dists, matching_indices).mean(-1)
            # 最终置信得分
            confidence_scores = confidence_scores * compatible_scores.unsqueeze(-1)
            ref_token_indices = self.seeding(compatible_scores, confidence_scores)
            ref_spot_mask, ref_spot_indices = self.spot_guided_attentions[i].select_spots(
                ref_idx[..., :self.k+1], src_idx[..., :self.spot_k], confidence_scores, matching_indices, self.spots
            )
            ref_compatibility.append(compatible_scores)
            
            confidence_scores, matching_indices = torch.max(matching_scores.transpose(1, 2), dim=-1, keepdim=True)
            compatible_scores = self.compatibility_scores(src_dists, ref_dists, matching_indices).mean(-1)
            confidence_scores = confidence_scores * compatible_scores.unsqueeze(-1)
            src_token_indices = self.seeding(compatible_scores, confidence_scores)
            src_spot_mask, src_spot_indices = self.spot_guided_attentions[i].select_spots(
                src_idx[..., :self.k+1], ref_idx[..., :self.spot_k], confidence_scores, matching_indices, self.spots
            )
            src_compatibility.append(compatible_scores)

            ref_feats = self.consistency_aware_attentions[i](
                ref_feats, ref_feats, ref_token_indices.unsqueeze(1)
            )
            src_feats = self.consistency_aware_attentions[i](
                src_feats, src_feats, src_token_indices.unsqueeze(1)
            )

            new_ref_feats = self.spot_guided_attentions[i](
                ref_feats, src_feats, ref_spot_indices, attention_mask=ref_spot_mask
            )
            new_src_feats = self.spot_guided_attentions[i](
                src_feats, ref_feats, src_spot_indices, attention_mask=src_spot_mask
            )
        
        ref_compatibility = torch.stack(ref_compatibility, dim=-1)
        src_compatibility = torch.stack(src_compatibility, dim=-1)
        return new_ref_feats, new_src_feats, correlation, ref_compatibility, src_compatibility