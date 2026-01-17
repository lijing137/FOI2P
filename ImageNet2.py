import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.checkpoint as checkpoint
from typing import Optional


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], + (1,) * (x.ndim - 1))
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor()
    output = x.div(keep_prob)
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


# 4*4像素为一个patch，所以可以理解为下采样4倍
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_c = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity

    def forward(self, x):
        _, _, H, W = x.shape
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # left right top bottom front back
            x = F.pad(x, (0, self.patch_size[1] - H % self.patch_size[1],
                          0, self.patch_size[0] - W % self.patch_size[0],
                          0, 0))

        # 下采样
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten  [B C H W] -> [B C HW]
        # transpose [B C HW] -> [B HW C]
        x = x.flatten(2).transpose(1, 2)
        # 将通道维度修改为C=96，增加特征
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        # 重新排列张量形状，-1为自动判断size
        x = x.view(B, -1, 4 * C)
        # 特征归一化处理，方便网络训练
        x = self.norm(x)
        # 特征降维
        x = self.reduction(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# 这部分的作用为shift后为不是连续区域增加mask
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # 这个scale就是 1/sqrt(d)
        self.scale = head_dim ** -0.5
        # 定义参数列表（相对位置变换）
        # 不太懂
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[0] - 1), num_heads))
        corrds_h = torch.arange(self.window_size[0])
        corrds_w = torch.arange(self.window_size[1])
        # stack表示把行索引和列索引叠加后就得到了（0，0）.......（3，3）
        coords = torch.stack(torch.meshgrid([corrds_h, corrds_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)  # [2,Mh*Mw]
        # 广播后相减，求相对位置，然后展平，为什么要相减
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2,Mh*Mw,Mh*Mw]
        # 为啥要这么干
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # [batch_size*num_windows,Mh*Mw,total_embed_dim]
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        # 对最后两个维度进行转职
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask [nW,Mh*Mw,Mh*Mw]
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# 将图片分割为一个个Window大小的图片
def window_partition(x, window_size: int):
    B, H, W, C = x.shape(x)
    # 将图片按照窗口分割后，将多个字图片视为Batch
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute:[B,H//M,M,W//M,M,C] -> [B,H//M,W//M,M,M,C]
    # view: [B,H//M,W//M,M,M,C] -> [B*num_windows,M,M,C]
    # 将permute后的数据变成连续的数据，方便后续View
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


# 讲一个个Window恢复成大图片
def window_reverse(windows, window_size: int, H: int, W: int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0, norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        # 除法，向下取整
        self.shift_size = window_size // 2

        # build blocks
        # 定一个包含多个swin transformer block的模块列表
        # 用于生成包含‘depth’个swin transformer block
        # 就是每个stage中的swin transformer
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                # 如果整除，shift_size=0；如果不能整除shift_size=window_size//2
                # 用于判断使用w-msa还是sw-sma
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)
        ])

        # patch merging layer
        # 导入downsample 参数
        # downsample就是patchmerging
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    # 为sw-msa添加attention mask
    def create_mask(self, x, H, W):
        # 保证Hp和Hw是window_size的整数倍，保证图片倍Windows size整除
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 创建一个全0的张量，为了后面应用mask
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))

        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        # 将输入张量切分成多个不重叠窗口【1 Hp Wp 1】-》 【nW Mh Mw 1】
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # 【nW Mh*Mw】
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

        # shift window后图片不连续，生成attn_mask把不连续的部分设置为-100
        attn_mask = attn_mask.mask_fill(attn_mask != 0, float(-100.0)).mask_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        attn_mask = self.create_mask(x, H, W)  # [nW,Mh*Mw,Mh*Mw]
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        # swin transformer block的流程就是
        # LayerNorm -> attention -> dropout -> LayerNorm -> MLP -> dropout
        self.norm1 = norm_layer(dim)
        # w-sma sw-msa
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size),
                                    num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                    proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.qkv_bias=qkv_bias
        # self.drop=drop

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 把feature map 给pad到window的整数倍
        pad_l = pad_t = 0
        # 后面的取余是必要的，他保证在feature map是window_size整数倍时，不需要补0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        # 大于0，sw-sma；小于0，w-sma
        # 此处先进行w-sma的描述，下面还会描述sw-sma
        if self.shift_size > 0:
            # 将第一维和第二维分别向上，向左平移一位
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # 窗口移完后需要重新分割图片
        # 划分窗口
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B,Mh,Mw,C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B,Mh*Mw,C]

        # w-msa  sw-msa
        # 这个没明白在干什么，好像是在遮蔽不连续的部分，
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # 拼接patch为完整图片
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B,Mh,Mw,C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B,H',W',C]

        # 拼接完整图片后再位移才是原始
        if self.shift_size > 0:
            # 将第一维和第二维分别向上，向左平移一位
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        # 这一部分顶后面所有norm+mlp+drop
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SwinTransformer(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels    2^3=8
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # 分割图片为不重叠的patch
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        # 逐渐增加,随即将输入的一部分设置为0，防止过拟合
        # p为丢弃的概率，每个元素都有p的概率被设置为0
        self.pos_drop = nn.Dropout(p=drop_rate)
        # stochastic depth
        # 生成一系列0-drop_path_rate之间均匀分布的值，控制每一层丢弃的概率
        # 数量为sum(depths)，在此处为2+2+6+2=12
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths(i_layer),
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)  # 关闭后效率变高
            # 将四个阶段的layers添加到一起
            self.layers.append(layers)

        # 根据自己的网络，此处可能会输出到align中进行对齐
        # 对于分类网络，后面会添加Layer Norm层，全局池化层，全连接层得到最终输出
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity

        self.apply(self._init_weights)

    def _init_weight(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)  # 【B L C】
        # 将张量x的二维和三维交换 [B L C] -> [B C L] -> [B C 1]
        # 对第三个维度进行池化
        x = self.avgpool(x.transpose(1, 2))
        # 按第二个维度展平
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


def swin_tiny_patch4_window7_224(num_classes: int = 1000, **kwargs):
    ImageTransformer = SwinTransformer(in_chans=3,
                                       patch_size=4,
                                       window_size=7,
                                       embed_dim=96,
                                       depths=(2, 2, 6, 2),
                                       num_heads=(3,6,12,24),
                                       num_classes=num_classes,
                                       **kwargs)
    return ImageTransformer
