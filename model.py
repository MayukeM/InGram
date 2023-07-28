import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


class InGramEntityLayer(nn.Module):  # 实体层，输入是实体嵌入，关系嵌入，三元组，输出是实体嵌入
    def __init__(self, dim_in_ent, dim_out_ent, dim_rel, bias=True, num_head=8):
        super(InGramEntityLayer, self).__init__()

        self.dim_out_ent = dim_out_ent  # 输出维度 256
        self.dim_hid_ent = dim_out_ent // num_head  # 隐藏层维度，等于输出维度除以头数 256/8=32
        assert dim_out_ent == self.dim_hid_ent * num_head  # 确保输出维度是隐藏层维度的整数倍，assert作用是断言，如果后面的表达式为False，就会抛出AssertionError的异常
        self.num_head = num_head  # 头数 8

        self.attn_proj = nn.Linear(2 * dim_in_ent + dim_rel, dim_out_ent, bias=bias)  # 用于计算注意力权重， Attention Projection 注意力投影层
        self.attn_vec = nn.Parameter(torch.zeros((1, num_head, self.dim_hid_ent)))  # 参数，用于计算注意力权重
        self.aggr_proj = nn.Linear(dim_in_ent + dim_rel, dim_out_ent, bias=bias)  # 用于聚合邻居信息

        self.dim_rel = dim_rel  # 关系嵌入维度，256
        self.act = nn.LeakyReLU(negative_slope=0.2)  # 激活函数, LeakyReLU是一个带泄露参数的ReLU函数，当输入小于0时，LeakyReLU与ReLU相比，LeakyReLU会有0.2的梯度，而不是0
        self.bias = bias  # 是否使用偏置
        self.param_init()  # 参数初始化

    def param_init(self):
        nn.init.xavier_normal_(self.attn_proj.weight, gain=nn.init.calculate_gain('relu'))  # 参数初始化，xavier_normal_是一种参数初始化方法，gain是增益系数
        nn.init.xavier_normal_(self.attn_vec, gain=nn.init.calculate_gain('relu'))  # 参数初始化
        nn.init.xavier_normal_(self.aggr_proj.weight, gain=nn.init.calculate_gain('relu'))  # 参数初始化
        if self.bias:  # 如果使用偏置
            nn.init.zeros_(self.attn_proj.bias)  # 将偏置初始化为0，zeros_是一种参数初始化方法
            nn.init.zeros_(self.aggr_proj.bias)  # 将偏置初始化为0

    def forward(self, emb_ent, emb_rel, triplets):
        num_ent = len(emb_ent)
        num_rel = len(emb_rel)
        head_idxs = triplets[..., 0]
        rel_idxs = triplets[..., 1]
        tail_idxs = triplets[..., 2]

        ent_freq = torch.zeros((num_ent,)).cuda().index_add(dim=0, index=tail_idxs, \
                                                            source=torch.ones_like(tail_idxs,
                                                                                   dtype=torch.float).cuda()).unsqueeze(
            dim=1)

        self_rel = torch.zeros((num_ent, self.dim_rel)).cuda().index_add(dim=0, index=tail_idxs,
                                                                         source=emb_rel[rel_idxs]) / ent_freq

        # add self-loops
        emb_rels = torch.cat([emb_rel[rel_idxs], self_rel], dim=0)
        head_idxs = torch.cat([head_idxs, torch.arange(num_ent).cuda()], dim=0)
        tail_idxs = torch.cat([tail_idxs, torch.arange(num_ent).cuda()], dim=0)

        concat_mat_att = torch.cat([emb_ent[tail_idxs], emb_ent[head_idxs], \
                                    emb_rels], dim=-1)

        attn_val_raw = (self.act(self.attn_proj(concat_mat_att).view(-1, self.num_head, self.dim_hid_ent)) *
                        self.attn_vec).sum(dim=-1, keepdim=True)

        scatter_idx = tail_idxs.unsqueeze(dim=-1).repeat(1, self.num_head).unsqueeze(dim=-1)

        attn_val_max = torch.zeros((num_ent, self.num_head, 1)).cuda().scatter_reduce(dim=0, \
                                                                                      index=scatter_idx, \
                                                                                      src=attn_val_raw, reduce='amax', \
                                                                                      include_self=False)
        attn_val = torch.exp(attn_val_raw - attn_val_max[tail_idxs])

        attn_sums = torch.zeros((num_ent, self.num_head, 1)).cuda().index_add(dim=0, index=tail_idxs, source=attn_val)

        beta = attn_val / (attn_sums[tail_idxs] + 1e-16)

        concat_mat = torch.cat([emb_ent[head_idxs], emb_rels], dim=-1)

        aggr_val = beta * self.aggr_proj(concat_mat).view(-1, self.num_head, self.dim_hid_ent)

        output = torch.zeros((num_ent, self.num_head, self.dim_hid_ent)).cuda().index_add(dim=0, index=tail_idxs,
                                                                                          source=aggr_val)

        return output.flatten(1, -1)


class InGramRelationLayer(nn.Module):  # 用于计算关系嵌入,输入为关系嵌入和三元组
    def __init__(self, dim_in_rel, dim_out_rel, num_bin, bias=True, num_head=8):
        super(InGramRelationLayer, self).__init__()

        self.dim_out_rel = dim_out_rel
        self.dim_hid_rel = dim_out_rel // num_head
        assert dim_out_rel == self.dim_hid_rel * num_head

        self.attn_proj = nn.Linear(2 * dim_in_rel, dim_out_rel, bias=bias)
        self.attn_bin = nn.Parameter(torch.zeros(num_bin, num_head, 1))
        self.attn_vec = nn.Parameter(torch.zeros(1, num_head, self.dim_hid_rel))
        self.aggr_proj = nn.Linear(dim_in_rel, dim_out_rel, bias=bias)
        self.num_head = num_head

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.num_bin = num_bin
        self.bias = bias

        self.param_init()

    def param_init(self):
        nn.init.xavier_normal_(self.attn_proj.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.attn_vec, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.aggr_proj.weight, gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.zeros_(self.attn_proj.bias)
            nn.init.zeros_(self.aggr_proj.bias)

    def forward(self, emb_rel, relation_triplets):
        num_rel = len(emb_rel)

        head_idxs = relation_triplets[..., 0]
        tail_idxs = relation_triplets[..., 1]
        concat_mat = torch.cat([emb_rel[head_idxs], emb_rel[tail_idxs]], dim=-1)

        attn_val_raw = (self.act(self.attn_proj(concat_mat).view(-1, self.num_head, self.dim_hid_rel)) * \
                        self.attn_vec).sum(dim=-1, keepdim=True) + self.attn_bin[relation_triplets[..., 2]]

        scatter_idx = head_idxs.unsqueeze(dim=-1).repeat(1, self.num_head).unsqueeze(dim=-1)

        attn_val_max = torch.zeros((num_rel, self.num_head, 1)).cuda().scatter_reduce(dim=0, \
                                                                                      index=scatter_idx, \
                                                                                      src=attn_val_raw, reduce='amax', \
                                                                                      include_self=False)
        attn_val = torch.exp(attn_val_raw - attn_val_max[head_idxs])

        attn_sums = torch.zeros((num_rel, self.num_head, 1)).cuda().index_add(dim=0, index=head_idxs, source=attn_val)

        beta = attn_val / (attn_sums[head_idxs] + 1e-16)

        output = torch.zeros((num_rel, self.num_head, self.dim_hid_rel)).cuda().index_add(dim=0, \
                                                                                          index=head_idxs,
                                                                                          source=beta * self.aggr_proj(
                                                                                              emb_rel[tail_idxs]).view(
                                                                                              -1, self.num_head,
                                                                                              self.dim_hid_rel))

        return output.flatten(1, -1)


class InGram(nn.Module):
    def __init__(self, dim_ent, hid_dim_ratio_ent, dim_rel, hid_dim_ratio_rel, num_bin, num_layer_ent=2,
                 num_layer_rel=2, \
                 num_head=8, bias=True):
        super(InGram, self).__init__()

        layers_ent = []  # 用于计算实体嵌入
        layers_rel = []  # 用于计算关系嵌入
        layer_dim_ent = hid_dim_ratio_ent * dim_ent  # 实体嵌入的维度=隐藏层的维度*实体维度 8*32=256
        layer_dim_rel = hid_dim_ratio_rel * dim_rel  # 关系嵌入的维度=隐藏层的维度*关系维度 4*32=128
        for _ in range(num_layer_ent):  # 用于计算实体嵌入
            layers_ent.append(InGramEntityLayer(layer_dim_ent, layer_dim_ent, layer_dim_rel, \
                                                bias=bias, num_head=num_head))  # 实体嵌入的维度
        for _ in range(num_layer_rel):
            layers_rel.append(InGramRelationLayer(layer_dim_rel, layer_dim_rel, num_bin, \
                                                  bias=bias, num_head=num_head))
        res_proj_ent = []  # 用于计算实体嵌入
        for _ in range(num_layer_ent):
            res_proj_ent.append(nn.Linear(layer_dim_ent, layer_dim_ent, bias=bias))

        res_proj_rel = []
        for _ in range(num_layer_rel):
            res_proj_rel.append(nn.Linear(layer_dim_rel, layer_dim_rel, bias=bias))

        self.res_proj_ent = nn.ModuleList(res_proj_ent)
        self.res_proj_rel = nn.ModuleList(res_proj_rel)
        self.bias = bias
        self.ent_proj1 = nn.Linear(dim_ent, layer_dim_ent, bias=bias)
        self.ent_proj2 = nn.Linear(layer_dim_ent, dim_ent, bias=bias)
        self.layers_ent = nn.ModuleList(layers_ent)
        self.layers_rel = nn.ModuleList(layers_rel)

        self.rel_proj1 = nn.Linear(dim_rel, layer_dim_rel, bias=bias)
        self.rel_proj2 = nn.Linear(layer_dim_rel, dim_rel, bias=bias)
        self.rel_proj = nn.Linear(dim_rel, dim_ent, bias=bias)
        self.num_layer_ent = num_layer_ent
        self.num_layer_rel = num_layer_rel
        self.act = nn.ReLU()

        self.param_init()

    def param_init(self):
        nn.init.xavier_normal_(self.ent_proj1.weight, gain=nn.init.calculate_gain('relu'))  # 初始化权重
        nn.init.xavier_normal_(self.ent_proj2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.rel_proj1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.rel_proj2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.rel_proj.weight, gain=nn.init.calculate_gain('relu'))
        for layer_idx in range(self.num_layer_ent):
            nn.init.xavier_normal_(self.res_proj_ent[layer_idx].weight, gain=nn.init.calculate_gain('relu'))
        for layer_idx in range(self.num_layer_rel):
            nn.init.xavier_normal_(self.res_proj_rel[layer_idx].weight, gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.zeros_(self.ent_proj1.bias)  # 初始化偏置
            nn.init.zeros_(self.ent_proj2.bias)
            nn.init.zeros_(self.rel_proj1.bias)
            nn.init.zeros_(self.rel_proj2.bias)
            nn.init.zeros_(self.rel_proj.bias)
            for layer_idx in range(self.num_layer_ent):
                nn.init.zeros_(self.res_proj_ent[layer_idx].bias)
            for layer_idx in range(self.num_layer_rel):
                nn.init.zeros_(self.res_proj_rel[layer_idx].bias)

    def forward(self, emb_ent, emb_rel, triplets, relation_triplets):

        layer_emb_ent = self.ent_proj1(emb_ent)
        layer_emb_rel = self.rel_proj1(emb_rel)

        for layer_idx, layer in enumerate(self.layers_rel):
            layer_emb_rel = layer(layer_emb_rel, relation_triplets) + \
                            self.res_proj_rel[layer_idx](layer_emb_rel)
            layer_emb_rel = self.act(layer_emb_rel)

        for layer_idx, layer in enumerate(self.layers_ent):
            layer_emb_ent = layer(layer_emb_ent, layer_emb_rel, triplets) + \
                            self.res_proj_ent[layer_idx](layer_emb_ent)
            layer_emb_ent = self.act(layer_emb_ent)

        return self.ent_proj2(layer_emb_ent), self.rel_proj2(layer_emb_rel)

    def score(self, emb_ent, emb_rel, triplets):

        head_idxs = triplets[..., 0]
        rel_idxs = triplets[..., 1]
        tail_idxs = triplets[..., 2]
        head_embs = emb_ent[head_idxs]
        tail_embs = emb_ent[tail_idxs]
        rel_embs = self.rel_proj(emb_rel[rel_idxs])
        output = (head_embs * rel_embs * tail_embs).sum(dim=-1)
        return output
