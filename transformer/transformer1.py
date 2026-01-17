import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points, knn_gather
from scipy.spatial.distance import correlation

from transformer.transformer import *
from CASTmodel.cast.spot_attention import SpotTransformerLayer, Upsampling
from imagenet import ImgUpSample

class SpotGuidedAggregation(nn.Module):
    def __init__(self,
                 input_dim_l,
                 input_dim_h,
                 hidden_dim,
                 num_heads,
                 dropout,
                 activation_fn,
                 blocks,
                 down_k,
                 dual_normalization
                 ):
        super(SpotGuidedAggregation, self).__init__()

        self.input_dim_l = input_dim_l
        self.input_dim_h = input_dim_h
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.blocks = blocks
        self.down_k = down_k
        self.dual_normalization = dual_normalization

        self.im_upsamp_final = ImgUpSample(128)
        self.pc_upsamp_final = Upsampling(128, 128)

        self.self_attentions18 = nn.ModuleList()
        self.cross_attentions18 = nn.ModuleList()
        self.spot_guided_attentions18 = nn.ModuleList()

        # 循环模块
        for _ in range(self.blocks):

            self.self_attentions18.append(LocalFeatureTransformer(D_MODEL=128, NHEAD=4, LAYER_NAMES=['self'] * 1,
                                                                     ATTENTION='full'))
            self.cross_attentions18.append(LocalFeatureTransformer(D_MODEL=128, NHEAD=4, LAYER_NAMES=['cross'] * 1,
                                                                      ATTENTION='full'))
            self.spot_guided_attentions18.append(SpotTransformerLayer(128, self.num_heads, False, self.dropout,
                                                                      self.activation_fn))

    def matching_scores(self, input_states: torch.Tensor, memory_states: torch.Tensor, dual_normalization: bool = True):
        input_states = F.normalize(input_states, dim=-1)
        memory_states = F.normalize(memory_states, dim=-1)
        if input_states.ndim == 2:
            matching_scores = torch.einsum('mc,nc->mn', input_states, memory_states) / 0.1
        else:
            matching_scores = torch.einsum('bmc,bnc->bmn', input_states, memory_states) / 0.1
        # matching_scores = F.softmax(matching_scores, -1) * F.softmax(matching_scores, -2)
        return matching_scores

    def im_similarity_scores(self, im_feats_h: torch.Tensor, k):
        B, C, H, W = im_feats_h.shape
        P = H * W  # 图像像素总数
        # 使用unfold提取每个像素的邻域特征（包含中心像素）, 边界区域补0,先用常数填充到四周
        im_feats_padded = F.pad(im_feats_h, (k // 2, k // 2, k // 2, k // 2), mode='replicate')
        # im_feats_padded = F.pad(im_feats_h, (k // 2, k // 2, k // 2, k // 2), mode='constant', value=1e-8)
        # 再展开。因为已经手动 pad 了，这里 padding=0
        patches = F.unfold(im_feats_padded, kernel_size=(k, k), padding=0, stride=1)
        # 调整形状以分离邻域位置, 方便按 位置 idx 索引
        patches = patches.view(B, C, k * k, P)
        # 计算每个像素与邻居的特征内积相似度
        neigh = im_feats_h.reshape(B, C, P).unsqueeze(2)
        sim_score = (F.normalize(patches, dim=1) * F.normalize(neigh, dim=1)).sum(dim=1)
        sim_score = F.softmax(sim_score.view(B, k * k, H, W), dim=1)
        return sim_score

    def pairwise_scores(self, x: torch.Tensor, y: torch.Tensor, chunk: int = 640,
                        dual_normalization: bool = True):
        B, M, C = x.shape
        max_vals = []
        max_inds = []
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        # 逐块遍历查询向量
        for i in range(0, M, chunk):
            x_blk = x[:, i:i + chunk]  # (B, chunk_size, C)
            # (B, chunk_size, N)  计算点积相似度
            sim_blk = torch.einsum('bmc,bnc->bmn', x_blk, y)
            # 取该块中每个查询的最大得分及索引
            v_blk, idx_blk = sim_blk.max(dim=-1)  # (B, chunk_size)
            max_vals.append(v_blk)
            max_inds.append(idx_blk)

        max_vals = torch.cat(max_vals, dim=1)  # (B, M)
        max_inds = torch.cat(max_inds, dim=1)  # (B, M)
        # 沿查询维度拼回完整结果
        return max_vals, max_inds

    def im_select_score(self, best_score_map: torch.Tensor, sim_score: torch.Tensor, expand_num):
        B, C, H, W = 1, 128, 20, 64
        P = H * W  # 图像像素总数
        # 先用常数填充到四周
        # score_map_padded = F.pad(best_score_map.unsqueeze(1),
        #                          (expand_num // 2, expand_num // 2, expand_num // 2, expand_num // 2),
        #                          mode='constant',
        #                          value=1e-8)
        # 此处best_score_map进行pad，但是它是否和图像边界一致呢？
        score_map_padded = F.pad(best_score_map.unsqueeze(1),
                                 (expand_num // 2, expand_num // 2, expand_num // 2, expand_num // 2),
                                 mode='replicate')
        # 再展开。因为已经手动 pad 了，这里 padding=0
        score_patches = F.unfold(score_map_padded, kernel_size=(expand_num, expand_num), padding=0, stride=1)
        conf_score = score_patches.view(B, expand_num * expand_num, H, W)
        # 元素逐位相乘，得到选择得分 (sim_score * conf_score)
        select_score = sim_score * conf_score
        # 为了确保边界上不存在无效邻居被选中，将填充区域的 select_score 置为一个很小的值（-inf）
        index_grid = torch.arange(P, dtype=torch.long, device=select_score.device).view(1, 1, H, W)
        # 将超出图像范围的索引值设为 -1
        index_grid_pad = F.pad(index_grid.float(),
                               pad=(expand_num // 2, expand_num // 2, expand_num // 2, expand_num // 2),
                               mode='constant', value=-1.0)
        idx_patches = F.unfold(index_grid_pad, kernel_size=expand_num)
        idx_patches = idx_patches.long()
        # 找到越界的邻居位置并赋值 -1e8 neighbor_idx是周围24个像素的索引值
        invalid_mask = (idx_patches == -1).reshape(B, -1, H, W)  # True 表示该邻居越界无效
        select_score[invalid_mask] = float('-1e8')  # 越界邻居选择得分设为负无穷
        return select_score, idx_patches

    def im_seeding(self, select_score: torch.Tensor, neighbor_idx: torch.Tensor, best_index_map: torch.Tensor, spot_num,
                   expand_num):
        B, H, W = 1, 20, 64
        # 将 select_score 的中心位置（即自身像素）置为一个很小的值，防止被选中
        select_score[:, (expand_num * expand_num) // 2, :, :] = -1e8  # 防止topk报错
        # topk_idx 形状 (B, 3, H, W), 包含每个像素Top3邻居的序号
        topk_values, topk_idx = torch.topk(select_score, k=spot_num - 1, dim=1)
        # 用于选择ref中心像素，值为(expand_num * expand_num) // 2
        extra = torch.full((1, 1, 20, 64), (expand_num * expand_num) // 2, dtype=topk_idx.dtype, device=topk_idx.device)
        topk_idx = torch.cat([extra, topk_idx], dim=1).reshape(B, -1, H * W)
        # 将索引0-25转化为1280图像像素索引
        neighbor_pixels = neighbor_idx.gather(dim=1, index=topk_idx)  # 形状 (B, 3, H, W)，每个元素是对应邻居的图像像素索引
        # 从 best_index_map 中获取这些邻居像素各自匹配的最佳点云索引
        best_index_flat = best_index_map.reshape(B, -1)  # 展平成大小为 P 的向量
        # 利用高级索引从 best_index_flat 提取每个邻居像素的最佳点云索引，
        src = best_index_flat.unsqueeze(1)  # [1, 1, 1280]
        src = src.expand(-1, neighbor_pixels.size(1), -1)  # [1, 5, 1280]
        neighbor_best_idx1 = torch.gather(src, 2, neighbor_pixels.long())
        seed = neighbor_best_idx1.squeeze(0).permute(1, 0).reshape(H, W, -1)
        return seed

    def im_spoting(self, pc_feats_h: torch.Tensor, seed: torch.Tensor, points_h: torch.Tensor, k, neighbor_pcidx1):
        p1_features = pc_feats_h.squeeze(0)
        device = p1_features.device
        seed = seed.to(device).long()  # 确保 seed 为 LongTensor 类型用于索引
        neighbor_pcidx1 = neighbor_pcidx1.squeeze(0)
        # 3. 使用 torch.topk 寻找最小的8个距离（largest=False 表示距离小，sorted=True 保证按距离升序）
        expanded_neighbors = neighbor_pcidx1[seed]
        # 6. 将最后两个维度 (4 和 9) 合并，得到每个像素点长度为36的索引列表
        spot = expanded_neighbors.reshape(20, 64, -1)

        # 避免冗余索引
        B, N, M = pc_feats_h.shape[0], pc_feats_h.shape[1], pc_feats_h.shape[1]
        attn_mask = torch.zeros((B, N, M), device=spot.device)
        attn_mask.scatter_(-1, spot.unsqueeze(0).reshape(B, N, -1), 1.)
        spot_mask, spot_indices_redu = attn_mask.topk(spot.shape[-1])
        spot_indices_redu = spot_indices_redu.long()
        return spot_mask, spot_indices_redu

    def pc_seeding(self, select_pc_score: torch.Tensor, neighbor_pcidx: torch.Tensor, best_match_idx: torch.Tensor,
                   spot_num):

        select_pc_score[:, :, 0] = -1e8  # 防止topk报错
        _, topk_pcidx = torch.topk(select_pc_score, spot_num - 1, dim=2)
        topk_pcidx = topk_pcidx.squeeze()
        # 构造 [0, 1, 2, ..., 1279]
        extra = torch.full((1280, 1), 0, dtype=topk_pcidx.dtype, device=topk_pcidx.device)
        topk_pcidx = torch.cat([extra, topk_pcidx], dim=1)
        # 使用gather从neighbor_pc_idx中取出对应邻居索引
        pc_idx = torch.gather(neighbor_pcidx.squeeze(0), 1, topk_pcidx)  # 1280*4
        best_match_idx = best_match_idx.squeeze(0)  # 1280
        # 此处就是索引点云对应的像素位置 注意：是像素位置
        pc_seed = best_match_idx[pc_idx]  # 1*1280*4
        return pc_seed

    def pc_spoting(self, pc_seed: torch.Tensor, expand_num, spot_num):
        H, W = 20, 64
        # 将seed中的5120分解为由（rows，cols）组成
        rows = pc_seed // W
        cols = pc_seed % W
        device = pc_seed.device
        offsets = torch.arange(-((expand_num - 1) // 2), (expand_num + 1) // 2,
                               device=device)  # tensor([-3, -2, -1, 0, 1, 2, 3], device=device)

        # 计算expand_num*expand_num邻域行列索引
        rows_expand = rows.unsqueeze(-1).unsqueeze(-1) + offsets.view(1, 1, expand_num, 1)
        cols_expand = cols.unsqueeze(-1).unsqueeze(-1) + offsets.view(1, 1, 1, expand_num)
        rows_nei = rows_expand.repeat(1, 1, 1, expand_num).reshape(1280, spot_num, expand_num * expand_num)
        cols_nei = cols_expand.repeat(1, 1, expand_num, 1).reshape(1280, spot_num, expand_num * expand_num)
        # 边界 clamp
        rows_nei = rows_nei.clamp(0, H - 1)
        cols_nei = cols_nei.clamp(0, W - 1)

        # 计算扁平索引: new_idx = row*W + col
        spot_idx = rows_nei * W + cols_nei

        B = 1
        N = H * W
        spot_idx = spot_idx.unsqueeze(0).reshape(B, N, -1)  # 1*1280*49

        attn_mask = torch.zeros((B, N, N), device=spot_idx.device)
        attn_mask.scatter_(-1, spot_idx, 1.)
        spot_mask, spot_idx_redu = attn_mask.topk(spot_idx.shape[-1])
        spot_idx_redu = spot_idx_redu.long()
        return spot_idx_redu, spot_mask

    def pc_similarity_scores(self, pc_feats_h: torch.Tensor, neighbor_pcidx):
        B, N, C = pc_feats_h.shape

        # 将特征拿成 (B, N, 1, C) 方便广播
        src = pc_feats_h.unsqueeze(2)  # (B, N, 1, C)
        nei = torch.gather(
            pc_feats_h.unsqueeze(1).expand(-1, N, -1, -1),  # (B, N, N, C)
            dim=2,
            index=neighbor_pcidx.unsqueeze(-1).expand(-1, -1, -1, C)
        )  # (B, N, k, C)
        sim_pc_score = (F.normalize(src, dim=-1) * F.normalize(nei, dim=-1)).sum(-1)  # (B, N, k)
        sim_pc_score = F.softmax(sim_pc_score, dim=-1)
        return sim_pc_score

    def forward(self,
                image_data,
                pc_data_dict,
                im_feats_h,
                pc_feats_h,
                im_feats_l,
                pc_feats_l):

        points_data = pc_data_dict['points']
        points_h = points_data[-1].unsqueeze(0)
        Hl, Wl, Hh, Wh = 10, 32, 20, 64
        B, C = 1, 128
        expand_num, spot_num = 5, 5
        # 因此在代码结束时，输出特征为new_im_feats_l, new_im_feats_h
        _, neighbor_pcidx1, _ = knn_points(points_h, points_h, K=expand_num * expand_num, return_nn=True)
        spot_correlation = []

        for i in range(self.blocks):  # 循环多次
            # 实现图像、点云自注意力self-attention融合
            raw18_im_feats_SFAtt, raw18_pc_feats_SFAtt,_,_ = self.self_attentions18[i](im_feats_h,
                                                                                   pc_feats_h)

            im_feats_h_van, pc_feats_h_van, cross_attn_im2pc, cross_attn_pc2im = self.cross_attentions18[i](raw18_im_feats_SFAtt,
                                                                raw18_pc_feats_SFAtt)

            """image-point spot-guided attention"""
            with torch.no_grad():
                im_sim_score = self.im_similarity_scores(raw18_im_feats_SFAtt.permute(0, 2, 1).reshape(B, C, Hh, Wh),
                                                         expand_num)
                best_score_map, best_index_map = self.pairwise_scores(raw18_im_feats_SFAtt,
                                                                      raw18_pc_feats_SFAtt)  # 1*1280
                best_score_map = best_score_map.view(B, Hh, Wh)
                best_index_map = best_index_map.view(B, Hh, Wh)
                select_score, neighbor_idx = self.im_select_score(best_score_map, im_sim_score, expand_num)
                im_seed = self.im_seeding(select_score, neighbor_idx, best_index_map, spot_num, expand_num)
                spot_mask, spot_indices_redu = self.im_spoting(raw18_pc_feats_SFAtt, im_seed, points_h,
                                                               expand_num, neighbor_pcidx1)

            im_feats_h_spo, score_im, idx_im = self.spot_guided_attentions18[i](raw18_im_feats_SFAtt, raw18_pc_feats_SFAtt,
                                                                    spot_indices_redu, attention_mask=spot_mask)

            """point——image SpotGuided attention"""
            with torch.no_grad():
                sim_pc_score = self.pc_similarity_scores(raw18_pc_feats_SFAtt, neighbor_pcidx1)
                similarity_score, similarity_idx = self.pairwise_scores(raw18_pc_feats_SFAtt,
                                                                        raw18_im_feats_SFAtt)
                similarity_score = similarity_score.squeeze(0)
                conf_pc_score = similarity_score[neighbor_pcidx1]
                select_pc_score = sim_pc_score * conf_pc_score
                pc_seed = self.pc_seeding(select_pc_score, neighbor_pcidx1, similarity_idx, spot_num)
                spot_idx, attn_mask = self.pc_spoting(pc_seed, expand_num, spot_num)
            pc_feats_h_spo, score_pc, idx_pc = self.spot_guided_attentions18[i](raw18_pc_feats_SFAtt, raw18_im_feats_SFAtt,
                                                                    spot_idx, attention_mask=attn_mask)

            spot_matching = self.matching_scores(im_feats_h_spo,
                                                 pc_feats_h_spo)
            spot_correlation.append(spot_matching)

            w = 0.5
            im_feats_h = w * im_feats_h_van + (1.0 - w) * im_feats_h_spo
            pc_feats_h = w * pc_feats_h_van + (1.0 - w) * pc_feats_h_spo

            # print("*")
            # print(im_feats_h_van.detach().max().item())
            # print(im_feats_h_spo.detach().max().item())
            # print(pc_feats_h_van.detach().max().item())
            # print(pc_feats_h_spo.detach().max().item())
            # print("*")

            # im_feats_h = self.im_upsamp_final(im_feats_h_spo.permute(0, 2, 1).reshape(B, C, 20, 64),
            #                                   im_feats_h_van.permute(0, 2, 1).reshape(B, C, 20, 64))
            # pc_feats_h = self.pc_upsamp_final(pc_feats_h_spo,
            #                                   pc_feats_h_van,
            #                                   neighbor_pcidx1[:,:,0].unsqueeze(-1))
            # im_feats_h = im_feats_h.permute(0, 2, 3, 1).reshape(B, -1, 128)

        # # 将融合后的通道修改为与原来算法对齐
        # im_feats_h = im_feats_h.permute(0, 2, 1).reshape(B, C, 20, 64)
        # pc_feats_h = pc_feats_h.permute(0, 2, 1)  # 1*1280*128-------1*128*1280
        spot_matching = torch.stack([c.squeeze(0) for c in spot_correlation], dim=0)
        return im_feats_h, pc_feats_h, spot_matching, cross_attn_im2pc, cross_attn_pc2im, score_im, idx_im, score_pc, idx_pc
