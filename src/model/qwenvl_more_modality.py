'''
Author: PengJie pengjieb@mail.ustc.edu.cn
Date: 2025-06-12 17:31:48
LastEditors: PengJie pengjieb@mail.ustc.edu.cn
LastEditTime: 2025-06-26 19:44:20
FilePath: /Qwen2-VL-Finetune/src/model/qwenvl_more_modality.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE

'''
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import einsum
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, _CONFIG_FOR_DOC, Qwen2_5_VLDecoderLayer
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import QWEN2_5_VL_INPUTS_DOCSTRING, Qwen2_5_VLCausalLMOutputWithPast
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor, Qwen2_5_VLProcessorKwargs
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.image_utils import ImageInput, VideoInput
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs
from positional_encodings.torch_encodings import PositionalEncoding1D
import random

import numpy as np

# from transformers.models.phi3.modeling_phi3 import Phi3


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context = None):
        B, N, C = x.shape
        if context is None:
            context = x
        # print(x.shape, context.shape)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, context.shape[1], 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v  = kv.unbind(0) 

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = drop

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs)
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class QFormerEncoder(nn.Module):
    def __init__(self, feature_size, num_patches, embed_dim, attn_drop=0.25, drop = 0.25, num_heads=4, qkv_bias=False, mlp_ratio = 4, act_layer=nn.GELU):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(num_patches, embed_dim))

        self.context_norm = nn.LayerNorm(feature_size)
        self.context_proj = nn.Linear(feature_size, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)

        self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x:torch.Tensor):

        x = self.context_proj(self.context_norm(x))
        
        B, C, D = x.shape
        
        query = self.query_tokens.unsqueeze(0).repeat(B, 1, 1)
        x = query + self.attn(self.norm1(query), x)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class MultiLevelQFormerEncoder(nn.Module):
    def __init__(self, feature_size, num_patches, n_modality, embed_dim, 
                 attn_drop=0.25, drop = 0.25, num_heads=4, qkv_bias=False, mlp_ratio = 4, 
                 act_layer=nn.GELU, more_config={}):
        super().__init__()
        n_unite = num_patches // n_modality
        self.more_config = more_config
        if self.more_config['token_distribution'] == 'top-1':
            self.n_patch_list = [num_patches//2 if i < n_modality-1 else num_patches + num_patches//2 * (n_modality-1) for i in range(n_modality)]
        elif self.more_config['token_distribution'] == 'top-2':
            self.n_patch_list = [num_patches//2 if i < n_modality-2 else num_patches + num_patches//2 for i in range(n_modality)]
        elif self.more_config['token_distribution'] == 'top-3':
            self.n_patch_list = [n_unite if i==0 else num_patches + n_unite for i in range(n_modality)]
        elif self.more_config['token_distribution'] == 'decrease' and self.more_config['discrete_token_number']:
            self.n_patch_list = [n_unite + 2 * i * n_unite for i in range(n_modality)]
        elif self.more_config['token_distribution'] == 'decrease' and not self.more_config['discrete_token_number']:
            self.n_patch_list = [num_patches * n_modality for i in range(n_modality)]
        # self.query_tokens = nn.Parameter(torch.randn(num_patches, embed_dim))
        self.query_tokens = nn.Parameter(torch.randn(self.n_patch_list[-1], embed_dim))


        self.context_norm = nn.LayerNorm(feature_size)
        self.context_proj = nn.Linear(feature_size, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)

        self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # self.n_patch_list = [56, 24, 24, 24]

    def forward(self, x:torch.Tensor, rank):
        
        x = self.context_proj(self.context_norm(x))
        
        B, C, D = x.shape
        if not self.more_config['discrete_token_number']:
            query = self.query_tokens[:rank].unsqueeze(0).repeat(B, 1, 1)
        else:
            query = self.query_tokens[:self.n_patch_list[rank]].unsqueeze(0).repeat(B, 1, 1)
        x = query + self.attn(self.norm1(query), x)[0]
        x = x + self.mlp(self.norm2(x))
        return x



class MultiLevelCompression(nn.Module):
    # MultiLevel Token Leaner
    def __init__(self, feature_size, num_patches, n_modality, 
                 embed_dim, attn_drop=0.25, drop = 0.25, num_heads=4, qkv_bias=False, mlp_ratio = 4, 
                 act_layer=nn.GELU, origin_tokens = None, more_config = {}):
        super().__init__()
        n_unite = num_patches // n_modality
        # self.n_patch_list = [n_unite + 2 * i * n_unite for i in range(n_modality)]
        n_unite = num_patches // n_modality
        self.more_config = more_config
        if self.more_config['token_distribution'] == 'top-1':
            self.n_patch_list = [num_patches//2 if i < n_modality-1 else num_patches + num_patches//2 * (n_modality-1) for i in range(n_modality)]
        elif self.more_config['token_distribution'] == 'top-2':
            self.n_patch_list = [num_patches//2 if i < n_modality-2 else num_patches + num_patches//2 for i in range(n_modality)]
        elif self.more_config['token_distribution'] == 'top-3':
            self.n_patch_list = [n_unite if i==0 else num_patches + n_unite for i in range(n_modality)]
        elif self.more_config['token_distribution'] == 'decrease' and self.more_config['discrete_token_number']:
            self.n_patch_list = [n_unite + 2 * i * n_unite for i in range(n_modality)]
        elif self.more_config['token_distribution'] == 'decrease' and not self.more_config['discrete_token_number']:
            self.n_patch_list = [num_patches * n_modality for i in range(n_modality)]
        # self.query_tokens = nn.Parameter(torch.randn(num_patches, embed_dim))
        # self.query_tokens = nn.Parameter(torch.randn(self.n_patch_list[-1], embed_dim))


        self.context_norm = nn.LayerNorm(feature_size)
        self.context_proj = nn.Linear(feature_size, embed_dim)
        
        rank_mlp = {}
        n_origin_tokens = int(256 * 10) if origin_tokens is None else origin_tokens
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        # for i, n_patch in enumerate(self.n_patch_list):
        #     rank_mlp[f"{i}"] = Mlp(n_origin_tokens, mlp_hidden_dim, n_patch, act_layer=act_layer, drop=drop)
        self.rank_mlp = Mlp(n_origin_tokens, mlp_hidden_dim, self.n_patch_list[-1], act_layer=act_layer, drop=drop)
        
        # self.norm1 = nn.LayerNorm(embed_dim)
        # self.attn = Attention(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.norm2 = nn.LayerNorm(embed_dim)
        # mlp_hidden_dim = int(embed_dim * mlp_ratio)

        # self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x:torch.Tensor, rank):
        inputs = self.context_norm(x)
        x = self.context_proj(inputs)
        
        B, C, D = x.shape
        x = x.permute(0, 2, 1) # B, D, C
        x = self.rank_mlp(x)
        if not self.more_config['discrete_token_number']:
            attn = x[:,:,:rank]
        else:
            attn = x[:,:,:self.n_patch_list[rank]]
        # for i_r in self.rank_mlp:
        #     i_x = self.rank_mlp[f"{i_r}"](x)
        #     if f"{i_r}" == f"{rank}":
        #         attn= i_x
        return attn.permute(0, 2, 1)
        attn = attn.softmax(dim=-2)
        # print(inputs.shape, x.shape, attn.shape)
        # out = attn.permute(0, 2, 1)
        out = einsum("... d i, ... n d -> ... i d", attn, inputs)
        # x = x.permute(0, 2, 1)
        # print(out.shape, x.shape)
        return out

class LinearPoolingParameterFree(nn.Module):
    def __init__(self, feature_size, num_patches, n_modality, 
                 embed_dim, attn_drop=0.25, drop = 0.25, num_heads=4, qkv_bias=False, mlp_ratio = 4, 
                 act_layer=nn.GELU, more_config = {}):
        super().__init__()
        n_unite = num_patches // n_modality
        self.n_patch_list = [n_unite + 2 * i * n_unite for i in range(n_modality)]
        self.more_config = more_config
        if self.more_config['token_distribution'] == 'top-1':
            self.n_patch_list = [num_patches//2 if i < n_modality-1 else num_patches + num_patches//2 * (n_modality-1) for i in range(n_modality)]
        elif self.more_config['token_distribution'] == 'top-2':
            self.n_patch_list = [num_patches//2 if i < n_modality-2 else num_patches + num_patches//2 for i in range(n_modality)]
        elif self.more_config['token_distribution'] == 'top-3':
            self.n_patch_list = [n_unite if i==0 else num_patches + n_unite for i in range(n_modality)]
        elif self.more_config['token_distribution'] == 'decrease' and self.more_config['discrete_token_number']:
            self.n_patch_list = [n_unite + 2 * i * n_unite for i in range(n_modality)]
        elif self.more_config['token_distribution'] == 'decrease' and not self.more_config['discrete_token_number']:
            self.n_patch_list = [num_patches * n_modality for i in range(n_modality)]
        
    def forward(self, x:torch.Tensor, rank):
        if not self.more_config['discrete_token_number']:
            n_target= rank
        else:
            n_target = self.n_patch_list[rank]
        
        n_window_size = x.shape[1]//n_target
        
        split_list = x[:,:n_window_size*(n_target-1)].split([n_window_size for _ in range(n_target-1)], dim=1)
        x_remain = x[:,n_window_size*(n_target-1):]
        n_ret = []
        for item in split_list:
            n_ret.append(
                item.mean(dim=1, keepdim=True)
            )
        n_ret.append(x_remain.mean(dim=1,keepdim=True))
        return torch.cat(n_ret, dim=1)


@torch.no_grad()
def memory_efficient_kmeans_cosine(features, num_clusters, sample_ratio=0.3, 
                                  batch_size=1024, max_iter=100, tol=1e-4):
    N, D = features.shape
    device = features.device
    dtype = features.dtype
    
    features = F.normalize(features, p=2, dim=1)
    
    sample_size = max(num_clusters * 2, int(N * sample_ratio)) 
    sample_indices = torch.randperm(N, device=device)[:sample_size]
    sample_features = features[sample_indices]
    
    with torch.no_grad(): 
        indices = torch.randperm(sample_size, device=device)[:num_clusters]
        centers = sample_features[indices].clone()

        for _ in range(50):
            similarity = torch.matmul(sample_features, centers.t())
            labels_sample = torch.argmax(similarity, dim=1)

            for c in range(num_clusters):
                mask = (labels_sample == c)
                if mask.sum() > 0:
                    centers[c] = torch.mean(sample_features[mask], dim=0)
                    centers[c] = F.normalize(centers[c], p=2, dim=0)

    labels = torch.zeros(N, device=device, dtype=torch.long)
    num_batches = (N + batch_size - 1) // batch_size  # 计算总批次数
    
    for b in range(num_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, N)
        batch_features = features[start:end]

        similarity = torch.matmul(batch_features, centers.t())
        batch_labels = torch.argmax(similarity, dim=1)
        labels[start:end] = batch_labels

    prev_centers = centers.clone()
    
    for _ in range(max_iter):
        cluster_counts = torch.zeros(num_clusters, device=device, dtype=torch.float32)
        cluster_sums = torch.zeros(num_clusters, D, device=device, dtype=dtype)
        
        for b in range(num_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, N)
            batch_features = features[start:end]
            batch_labels = labels[start:end]

            for c in range(num_clusters):
                mask = (batch_labels == c)
                if mask.sum() > 0:
                    cluster_counts[c] += mask.sum()
                    cluster_sums[c] += torch.sum(batch_features[mask], dim=0)

        for c in range(num_clusters):
            if cluster_counts[c] > 0:
                centers[c] = cluster_sums[c] / cluster_counts[c]
                centers[c] = F.normalize(centers[c], p=2, dim=0)

            

        for b in range(num_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, N)
            batch_features = features[start:end]
            
            similarity = torch.matmul(batch_features, centers.t())
            batch_labels = torch.argmax(similarity, dim=1)
            labels[start:end] = batch_labels
        

        center_diff = torch.norm(centers - prev_centers, dim=1).mean()
        if center_diff < tol:
            break
        prev_centers = centers.clone()
    
    return labels, centers

class CosineSimilarityPruning(nn.Module):
    def __init__(self, feature_size, num_patches, n_modality, 
                 embed_dim, attn_drop=0.25, drop = 0.25, num_heads=4, qkv_bias=False, mlp_ratio = 4, 
                 act_layer=nn.GELU, more_config = {}):
        super().__init__()
        n_unite = num_patches // n_modality
        self.n_patch_list = [n_unite + 2 * i * n_unite for i in range(n_modality)]
        self.more_config = more_config
        if self.more_config['token_distribution'] == 'top-1':
            self.n_patch_list = [num_patches//2 if i < n_modality-1 else num_patches + num_patches//2 * (n_modality-1) for i in range(n_modality)]
        elif self.more_config['token_distribution'] == 'top-2':
            self.n_patch_list = [num_patches//2 if i < n_modality-2 else num_patches + num_patches//2 for i in range(n_modality)]
        elif self.more_config['token_distribution'] == 'top-3':
            self.n_patch_list = [n_unite if i==0 else num_patches + n_unite for i in range(n_modality)]
        elif self.more_config['token_distribution'] == 'decrease' and self.more_config['discrete_token_number']:
            self.n_patch_list = [n_unite + 2 * i * n_unite for i in range(n_modality)]
        elif self.more_config['token_distribution'] == 'decrease' and not self.more_config['discrete_token_number']:
            self.n_patch_list = [num_patches * n_modality for i in range(n_modality)]
        
    def forward(self, x:torch.Tensor, rank):
        if not self.more_config['discrete_token_number']:
            n_target=rank
        else:
            n_target = self.n_patch_list[rank]
        
        B, L, D = x.shape
        if n_target > L:
            return x
        
        cluster_labels = []
        cluster_centers = []
        
        for b in range(B):
            batch_tokens = x[b]
            
            labels, centers = memory_efficient_kmeans_cosine(
                batch_tokens,
                n_target,
                sample_ratio=0.1,
                batch_size=1024,
                max_iter=100,
                tol=1e-4
            )
            cluster_centers.append(centers)
        return torch.stack(cluster_centers)

def average_cosine_similarity(tokens, block_size=1024):

    B, L, D = tokens.shape
    device = tokens.device
    avg_sims = torch.zeros(B, device=device)
    
    norm_tokens = F.normalize(tokens, p=2, dim=2)  
    
    for b in range(B):
        seq = norm_tokens[b]  
        total_sim = 0.0
        total_pairs = 0
        
        if L < 2:
            avg_sims[b] = 1.0
            continue
        
        num_blocks = (L + block_size - 1) // block_size
        
        for i_block in range(num_blocks):
            i_start = i_block * block_size
            i_end = min((i_block + 1) * block_size, L)
            current_block = seq[i_start:i_end]
            
            for j_block in range(i_block, num_blocks):
                j_start = j_block * block_size
                j_end = min((j_block + 1) * block_size, L)
                
                if i_block == j_block:
                    j_start = max(j_start, i_start + 1)
                    if j_start >= j_end:
                        continue
                    other_block = seq[j_start:j_end]  
                    sims = torch.matmul(current_block, other_block.T)  # (i_block_size, j_block_size)
                else:
                    other_block = seq[j_start:j_end]  # (j_block_size, D)
                    sims = torch.matmul(current_block, other_block.T)  # (i_block_size, j_block_size)
                
                total_sim += sims.sum()
                total_pairs += sims.numel()
        
        avg_sims[b] = 1 - total_sim / total_pairs if total_pairs > 0 else 0.0
    
    return avg_sims

class CosineSimilarityRedundancy(nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x:List[torch.Tensor],):

        cluster_results = []
        for item in x:
            cluster_results.append(
                average_cosine_similarity(item).unsqueeze(dim=-1)
            )
        return torch.cat(cluster_results, dim=-1) # B, M


class RandomOrder(nn.Module):
    def __init__(self):
        super().__init__()

        
    def forward(self, x:List[torch.Tensor],):

        # cluster_results = []
        n_modality = len(x)
        batch_size = x[0].shape[0]
        
        alpha = torch.ones(n_modality, device=x[0].device)
        dirichlet = torch.distributions.Dirichlet(alpha)
        batch_tensor = torch.stack([
            dirichlet.sample() for _ in range(batch_size)
        ])
        # print(batch_tensor)
        return batch_tensor # B, M

class Qwen2_5_VLProcessorOneToken(Qwen2_5_VLProcessor):
    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, n_frames = 4, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template, **kwargs)
        self.n_frames = n_frames
        
        
    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Qwen2_5_VLProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            Qwen2_5_VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.image_processor(images=None, videos=videos, **output_kwargs["images_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]

            fps = output_kwargs["videos_kwargs"].pop("fps", 2.0)
            if isinstance(fps, (int, float)):
                second_per_grid_ts = [self.image_processor.temporal_patch_size / fps] * len(video_grid_thw)
            elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
                second_per_grid_ts = [self.image_processor.temporal_patch_size / tmp for tmp in fps]
            else:
                raise ValueError(
                    f"The length of fps ({len(fps) if hasattr(fps, '__len__') else fps}) must be equal to the length of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number."
                )
            videos_inputs.update({"second_per_grid_ts": second_per_grid_ts})

        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>" * self.n_frames,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    text[i] = text[i].replace(
                        self.video_token,
                        "<|placeholder|>" * (video_grid_thw[index].prod() // merge_length),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)
        # print(len(text))
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})
        
def get_rank_order(input_tensor, descending=False):
    """
    Compute rank order for each row (along n_modality) in a tensor of shape (N, n_modality).
    
    Args:
        input_tensor (torch.Tensor): Input tensor with shape (N, n_modality).
        descending (bool): If True, rank from largest to smallest.
    
    Returns:
        torch.Tensor: Rank tensor with shape (N, n_modality), where each row contains ranks for that sample's modalities.
    """
    # Get sorted indices along each row (dim=1)
    sorted_indices = torch.argsort(input_tensor, dim=1, descending=descending)
    
    # Create a tensor of ranks (0 to n_modality-1) for each row
    n_modality = input_tensor.shape[1]
    ranks = torch.arange(n_modality, device=input_tensor.device).expand(input_tensor.shape[0], -1)
    
    # Initialize output tensor
    rank_order = torch.empty_like(sorted_indices)
    
    # Scatter ranks into correct positions using advanced indexing
    batch_indices = torch.arange(input_tensor.shape[0]).unsqueeze(1).expand(-1, n_modality)
    rank_order[batch_indices, sorted_indices] = ranks
    
    return rank_order


# class Phi3VLCausalGeneratioMore(Phi3VLCausalGeneratio):
#     def


def token_budgets_from_weight(weights, total_token_budgets):
    # weights: batch, n_modality
    batch_size, n_modality = weights.shape
    sum_weights = weights.sum(dim=1, keepdim=True)
    sum_weights = torch.where(sum_weights == 0, 
                             torch.tensor(n_modality, device=weights.device, dtype=sum_weights.dtype), 
                             sum_weights)
    
    base_budget = 2
    remaining_budget = total_token_budgets - base_budget * n_modality
    
    proportions = weights / sum_weights
    budgets_float = proportions * remaining_budget
    budgets_int = torch.floor(budgets_float).to(torch.int32)
    total_allocated = budgets_int.sum(dim=1)
    diff = remaining_budget - total_allocated
    # for i in range(batch_size):
    #     current_diff = max(0, diff[i].item())
    #     if diff[i] <= 0:
    #         continue
    #     fractional = budgets_float[i] - budgets_int[i].to(budgets_float.dtype)
    #     k = min(current_diff, n_modality)
    #     _, top_indices = torch.topk(fractional, k=k)
    #     budgets_int[i, top_indices] += 1
    final_budgets = budgets_int + base_budget
    final_budgets[:,-1] += diff
    # print(final_budgets.sum(dim=1), final_budgets)
    
    return final_budgets

class Qwen2_5_VLForConditionalGenerationMore(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config, more_config):
        super().__init__(config)
        self.config = config
        self.more_config = more_config
        # self.n_per_img = n_per_img
        # self.n_prefusion_layers = n_prefusion_layers
        # self.multilevel_qformer = config.multilevel_qformer
        # if not lazy_load:
        #     assign_qformer(self, modalities, multilevel_qformer)
        #     assign_prefusion(self, n_prefusion_layers)
        # print(n_image, n_depth, n_norm, n_flow, multilevel_qformer, n_prefusion_layers)
        self.total_multimodal_budgets = [self.more_config[mm] for mm in self.more_config['modality_list']]
        self.total_multimodal_budgets = sum(self.total_multimodal_budgets)
        assign_qformer(self, more_config)
        assign_prefusion(self, more_config)
    
    @add_start_docstrings_to_model_forward(QWEN2_5_VL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Qwen2_5_VLCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        n_image: Optional[torch.LongTensor] = None,
        depth_values: Optional[torch.Tensor] = None,
        norm_values: Optional[torch.Tensor] = None,
        flow_values: Optional[torch.Tensor] = None,
        norm_value_grid: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)

                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                depth_embeds = self.visual(depth_values, grid_thw=image_grid_thw)
                flow_embeds = self.visual(flow_values, grid_thw=image_grid_thw)
                norm_embeds = self.visual(norm_values, grid_thw=norm_value_grid)
                # print(image_embeds.shape, depth_embeds.shape, flow_embeds.shape, norm_embeds.shape)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                # print("n_image_tokens", n_image_tokens)
                
                n_batch = input_ids.shape[0]
                
                image_embeds = image_embeds.reshape(n_batch, -1, image_embeds.shape[-1])
                depth_embeds = depth_embeds.reshape(n_batch, -1, depth_embeds.shape[-1])
                flow_embeds = flow_embeds.reshape(n_batch, -1, flow_embeds.shape[-1])
                norm_embeds = norm_embeds.reshape(n_batch, -1, norm_embeds.shape[-1])

                if attention_mask is not None:
                    padding_mask = attention_mask.to(inputs_embeds.device)
                else:
                    padding_mask = torch.ones(n_batch, 0).to(inputs_embeds.device).bool()
                    # attention_mask = (input_ids > -1000000).to(torch.long).to(inputs_embeds.device).bool()
                non_fused_inputs_embeds = inputs_embeds
                global_image_features_length = image_embeds.size(1) + depth_embeds.size(1) + flow_embeds.size(1) + norm_embeds.size(1)
                # compressed_global_image_features_length = compressed_image_embeds.size(1) + compressed_depth_embeds.size(1) + compressed_flow_embeds.size(1) + compressed_norm_embeds.size(1)
                token_length_list = [image_embeds.size(1), depth_embeds.size(1), flow_embeds.size(1), norm_embeds.size(1), inputs_embeds.size(1)]
                x = torch.cat([image_embeds, depth_embeds, flow_embeds, norm_embeds,
                               inputs_embeds], dim=1)
                mask = torch.cat((torch.zeros((padding_mask.size(0),
                                                    global_image_features_length),device=padding_mask.device).bool(), padding_mask),dim=1)
                seq_len = mask.size(1)
                
                modality_rank = None
                if self.more_config['modality_ranker'] == 'attention':
                    if self.more_config['only_self_attention']:
                        lout = self.prefusion(x)
                        x = lout[0]
                        prefusion_attn = lout[1]
                        # print(x.shape, prefusion_attn.shape)
                        
                    elif self.more_config['learnable_attention'] and not self.more_config['only_self_attention']:
                        
                        prefusion_position_ids = (~mask).int().long().cumsum(-1) - 1
                        prefusion_position_ids.masked_fill_((~mask).int() == 0, 1)
                        seq_len = mask.size(1)
                        prefusion_attention_mask =(~mask).int()
                        prefusion_attention_mask = prefusion_attention_mask.unsqueeze(1)
                        prefusion_attention_mask = prefusion_attention_mask.unsqueeze(3)
                        prefusion_attention_mask = prefusion_attention_mask.expand(-1, -1, -1, seq_len)

                        if prefusion_position_ids.dim() == 2:
                            prefusion_position_ids = prefusion_position_ids[None, ...].expand(3, prefusion_position_ids.shape[0], -1)
                        prefusion_position_embeddings = self.model.rotary_emb(x, prefusion_position_ids)
                        # with torch.no_grad():
                        if hasattr(self, 'prefusion'): 
                            for layer in self.prefusion:
                                if self.is_gradient_checkpointing and self.training:
                                    lout = self.model._gradient_checkpointing_func(
                                        layer.__call__,
                                        x,
                                        prefusion_attention_mask,
                                        None, None, True, None, None, prefusion_position_embeddings
                                    )
                                else:
                                    lout = layer(x,
                                            attention_mask=prefusion_attention_mask, position_embeddings=prefusion_position_embeddings,
                                            output_attentions=True)
                                x = lout[0]
                                prefusion_attn = lout[1]
                        
                            
                    elif not self.more_config['learnable_attention'] and not self.more_config['only_self_attention']:
                        mask = torch.cat((torch.zeros((padding_mask.size(0),
                                                global_image_features_length),device=padding_mask.device).bool(), padding_mask),dim=1)
                        prefusion_position_ids = (~mask).int().long().cumsum(-1) - 1
                        prefusion_position_ids.masked_fill_((~mask).int() == 0, 1)
                        seq_len = mask.size(1)
                        prefusion_attention_mask =(~mask).int()
                        prefusion_attention_mask = prefusion_attention_mask.unsqueeze(1)
                        prefusion_attention_mask = prefusion_attention_mask.unsqueeze(3)
                        prefusion_attention_mask = prefusion_attention_mask.expand(-1, -1, -1, seq_len)

                        if prefusion_position_ids.dim() == 2:
                            prefusion_position_ids = prefusion_position_ids[None, ...].expand(3, prefusion_position_ids.shape[0], -1)
                        prefusion_position_embeddings = self.model.rotary_emb(x, prefusion_position_ids)
                        with torch.no_grad():
                            if hasattr(self, 'prefusion'): 
                                for layer in self.prefusion:
                                    if self.is_gradient_checkpointing and self.training:
                                        lout = self.model._gradient_checkpointing_func(
                                            layer.__call__,
                                            x,
                                            prefusion_attention_mask,
                                            None, None, True, None, None, prefusion_position_embeddings
                                        )
                                    else:
                                        lout = layer(x,
                                                attention_mask=prefusion_attention_mask, position_embeddings=prefusion_position_embeddings,
                                                output_attentions=True)
                                    x = lout[0]
                                    prefusion_attn = lout[1]
                        
                    prefusion_attn = prefusion_attn.sum(dim=-2).mean(dim=1) # batch size, seq_len
                    image_weight, depth_weight, flow_weight, norm_weight, text_weight = prefusion_attn.split(token_length_list, dim=1)
                    image_weight, depth_weight, flow_weight, norm_weight = image_weight.sum(-1, keepdim=True), depth_weight.sum(-1, keepdim=True), flow_weight.sum(-1, keepdim=True), norm_weight.sum(-1, keepdim=True)
                    modality_weight = torch.cat([image_weight, depth_weight, flow_weight, norm_weight], dim=1) # batch, n_modality
                    if self.more_config['discrete_token_number']:
                        modality_rank = get_rank_order(modality_weight)
                    else:
                        modality_rank = token_budgets_from_weight(modality_weight, self.total_multimodal_budgets)
                    n_batch, n_modality = modality_rank.shape
                elif self.more_config['modality_ranker'] == 'cosine_similarity':
                    image_embeds, depth_embeds, flow_embeds, norm_embeds, _ = x.split(token_length_list, dim=1)
                    modality_weight = self.prefusion([image_embeds, depth_embeds, flow_embeds, norm_embeds])
                    if self.more_config['discrete_token_number']:
                        modality_rank = get_rank_order(modality_weight)
                    else:
                        modality_rank = token_budgets_from_weight(modality_weight, self.total_multimodal_budgets)
                    # modality_rank = get_rank_order(modality_weight)
                    n_batch, n_modality = modality_rank.shape
                elif self.more_config['modality_ranker'] == 'random':
                    image_embeds, depth_embeds, flow_embeds, norm_embeds, _ = x.split(token_length_list, dim=1)
                    # print(x.shape)
                    modality_weight = self.prefusion([image_embeds, depth_embeds, flow_embeds, norm_embeds])
                    # modality_rank = get_rank_order(modality_weight)
                    
                    if self.more_config['discrete_token_number']:
                        modality_rank = get_rank_order(modality_weight)
                    else:
                        modality_rank = token_budgets_from_weight(modality_weight, self.total_multimodal_budgets)
                image_embeds, depth_embeds, flow_embeds, norm_embeds, _ = x.split(token_length_list, dim=1)
                # print(x.shape, prefusion_attn.shape)
                # exit()
                llm_input_image_embeds = []
                random_rank = [1,2,3,0]
                # print(modality_rank)
                
                # print(image_embeds.shape, depth_embeds.shape, flow_embeds.shape, norm_embeds.shape, modality_rank)
                # print(x.shape, non_fused_inputs_embeds.shape)
                # exit()
                # print(modality_rank)
                for bi in range(n_batch):
                    if modality_rank is None:
                        random.shuffle(random_rank)
                        image_rank, depth_rank, flow_rank, norm_rank = random_rank
                    else:
                        image_rank, depth_rank, flow_rank, norm_rank = modality_rank[bi]
            
                    compressed_image_embeds = self.m_qformer['image'](image_embeds[bi:bi+1], image_rank)
                    compressed_depth_embeds = self.m_qformer['depth'](depth_embeds[bi:bi+1], depth_rank)
                    compressed_flow_embeds = self.m_qformer['flow'](flow_embeds[bi:bi+1], flow_rank)
                    compressed_norm_embeds = self.m_qformer['norm'](norm_embeds[bi:bi+1], norm_rank)
                    llm_input_image_embeds.append(
                        torch.cat([compressed_image_embeds, compressed_depth_embeds, compressed_flow_embeds, compressed_norm_embeds], dim=1)
                    )
                llm_input_image_embeds = torch.cat(llm_input_image_embeds, dim=0)
                # fusion_text_features = x[:, -1 *input_ids.size(1):,:]
                # print(llm_input_image_embeds.shape)
                
                # fusion_text_features=fusion_text_features*(~padding_mask).unsqueeze(-1).int()+inputs_embeds*padding_mask.unsqueeze(-1)


                image_embeds = llm_input_image_embeds.reshape(-1, llm_input_image_embeds.shape[-1])
                n_image_features = image_embeds.shape[0]
                inputs_embeds = non_fused_inputs_embeds

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)
                # print(image_mask.shape)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                # print(image_mask.shape, image_embeds.shape, input_ids.shape, inputs_embeds.shape, mask.sum())
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)
        # print(attention_mask.shape, input_ids.shape)
        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # print(position_i ds.shape, attention_mask.shape, inputs_embeds.shape)
        # exit()
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
    
    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        # t, h, w = torch.tensor(10), torch.tensor(1), torch.tensor(1)
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas



class Qwen2_5_VLForConditionalGenerationMoreSA3D(Qwen2_5_VLForConditionalGenerationMore):
    def __init__(self, config, more_config):
        super().__init__(config, more_config)
        pos_model = PositionalEncoding1D(1408 // 3)
        x = torch.zeros(1, 256, 1408 // 3)
        self.pc_pos_embedding = pos_model(x).squeeze().cuda()
        self.pc_projector = nn.Sequential(
            *[nn.LayerNorm(1408), 
             Mlp(
                    1408, config.hidden_size * 2, config.hidden_size
                ),
             nn.LayerNorm(config.hidden_size), 
             ]
        )
    
    @add_start_docstrings_to_model_forward(QWEN2_5_VL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Qwen2_5_VLCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        n_image: Optional[torch.LongTensor] = None,
        depth_values: Optional[torch.Tensor] = None,
        norm_values: Optional[torch.Tensor] = None,
        pc_values: Optional[torch.Tensor] = None,
        pc_feature_values: Optional[torch.Tensor] = None,
        norm_value_grid: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)

                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                depth_embeds = self.visual(depth_values, grid_thw=image_grid_thw)
                norm_embeds = self.visual(norm_values, grid_thw=norm_value_grid)
                
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    pc_embeds = pc_feature_values
                    pc = pc_values.long()
                    all_pcs = torch.zeros((pc_embeds.shape))
                    for j in range(pc.shape[0]):
                        pcs = []
                        for i in range(3):
                            pc_i = pc[j][:, i]
                            pcs.append(self.pc_pos_embedding[pc_i])
                        pcs = torch.cat(pcs, -1)
                        all_pcs[j][:, :1407] = pcs
                    all_pcs = all_pcs.cuda()
                pc_embeds = pc_embeds + 0.01 * all_pcs
                # print(f"PC embedding:", pc_embeds.shape)
                pc_embeds = pc_embeds.to(dtype=image_embeds.dtype)
                pc_embeds = self.pc_projector(pc_embeds)
                
                
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                # print("n_image_tokens", n_image_tokens)
                
                n_batch = input_ids.shape[0]
                
                image_embeds = image_embeds.reshape(n_batch, -1, image_embeds.shape[-1])
                depth_embeds = depth_embeds.reshape(n_batch, -1, depth_embeds.shape[-1])
                pc_embeds = pc_embeds.reshape(n_batch, -1, depth_embeds.shape[-1])
                norm_embeds = norm_embeds.reshape(n_batch, -1, norm_embeds.shape[-1])

                if attention_mask is not None:
                    padding_mask = attention_mask.to(inputs_embeds.device)
                else:
                    padding_mask = torch.ones(n_batch, 0).to(inputs_embeds.device).bool()
                    # attention_mask = (input_ids > -1000000).to(torch.long).to(inputs_embeds.device).bool()
                non_fused_inputs_embeds = inputs_embeds
                global_image_features_length = image_embeds.size(1) + depth_embeds.size(1) + pc_embeds.size(1) + norm_embeds.size(1)
                # compressed_global_image_features_length = compressed_image_embeds.size(1) + compressed_depth_embeds.size(1) + compressed_flow_embeds.size(1) + compressed_norm_embeds.size(1)
                token_length_list = [image_embeds.size(1), depth_embeds.size(1), pc_embeds.size(1), norm_embeds.size(1), inputs_embeds.size(1)]
                x = torch.cat([image_embeds, depth_embeds, pc_embeds, norm_embeds,
                               inputs_embeds], dim=1)
                mask = torch.cat((torch.zeros((padding_mask.size(0),
                                                    global_image_features_length),device=padding_mask.device).bool(), padding_mask),dim=1)
                seq_len = mask.size(1)
                
                modality_rank = None
                if self.more_config['modality_ranker'] == 'attention':
                    if self.more_config['only_self_attention']:
                        lout = self.prefusion(x)
                        x = lout[0]
                        prefusion_attn = lout[1]
                        # print(x.shape, prefusion_attn.shape)
                        
                    elif self.more_config['learnable_attention'] and not self.more_config['only_self_attention']:
                        
                        prefusion_position_ids = (~mask).int().long().cumsum(-1) - 1
                        prefusion_position_ids.masked_fill_((~mask).int() == 0, 1)
                        seq_len = mask.size(1)
                        prefusion_attention_mask =(~mask).int()
                        prefusion_attention_mask = prefusion_attention_mask.unsqueeze(1)
                        prefusion_attention_mask = prefusion_attention_mask.unsqueeze(3)
                        prefusion_attention_mask = prefusion_attention_mask.expand(-1, -1, -1, seq_len)

                        if prefusion_position_ids.dim() == 2:
                            prefusion_position_ids = prefusion_position_ids[None, ...].expand(3, prefusion_position_ids.shape[0], -1)
                        prefusion_position_embeddings = self.model.rotary_emb(x, prefusion_position_ids)
                        # with torch.no_grad():
                        if hasattr(self, 'prefusion'): 
                            for layer in self.prefusion:
                                if self.is_gradient_checkpointing and self.training:
                                    lout = self.model._gradient_checkpointing_func(
                                        layer.__call__,
                                        x,
                                        prefusion_attention_mask,
                                        None, None, True, None, None, prefusion_position_embeddings
                                    )
                                else:
                                    lout = layer(x,
                                            attention_mask=prefusion_attention_mask, position_embeddings=prefusion_position_embeddings,
                                            output_attentions=True)
                                x = lout[0]
                                prefusion_attn = lout[1]
                        
                            
                    elif not self.more_config['learnable_attention'] and not self.more_config['only_self_attention']:
                        mask = torch.cat((torch.zeros((padding_mask.size(0),
                                                global_image_features_length),device=padding_mask.device).bool(), padding_mask),dim=1)
                        prefusion_position_ids = (~mask).int().long().cumsum(-1) - 1
                        prefusion_position_ids.masked_fill_((~mask).int() == 0, 1)
                        seq_len = mask.size(1)
                        prefusion_attention_mask =(~mask).int()
                        prefusion_attention_mask = prefusion_attention_mask.unsqueeze(1)
                        prefusion_attention_mask = prefusion_attention_mask.unsqueeze(3)
                        prefusion_attention_mask = prefusion_attention_mask.expand(-1, -1, -1, seq_len)

                        if prefusion_position_ids.dim() == 2:
                            prefusion_position_ids = prefusion_position_ids[None, ...].expand(3, prefusion_position_ids.shape[0], -1)
                        prefusion_position_embeddings = self.model.rotary_emb(x, prefusion_position_ids)
                        with torch.no_grad():
                            if hasattr(self, 'prefusion'): 
                                for layer in self.prefusion:
                                    if self.is_gradient_checkpointing and self.training:
                                        lout = self.model._gradient_checkpointing_func(
                                            layer.__call__,
                                            x,
                                            prefusion_attention_mask,
                                            None, None, True, None, None, prefusion_position_embeddings
                                        )
                                    else:
                                        lout = layer(x,
                                                attention_mask=prefusion_attention_mask, position_embeddings=prefusion_position_embeddings,
                                                output_attentions=True)
                                    x = lout[0]
                                    prefusion_attn = lout[1]
                        
                    prefusion_attn = prefusion_attn.sum(dim=-2).mean(dim=1) # batch size, seq_len
                    image_weight, depth_weight, pc_weight, norm_weight, text_weight = prefusion_attn.split(token_length_list, dim=1)
                    image_weight, depth_weight, pc_weight, norm_weight = image_weight.sum(-1, keepdim=True), depth_weight.sum(-1, keepdim=True), pc_weight.sum(-1, keepdim=True), norm_weight.sum(-1, keepdim=True)
                    modality_weight = torch.cat([image_weight, depth_weight, pc_weight, norm_weight], dim=1) # batch, n_modality
                    if self.more_config['discrete_token_number']:
                        modality_rank = get_rank_order(modality_weight)
                    else:
                        modality_rank = token_budgets_from_weight(modality_weight, self.total_multimodal_budgets)
                    n_batch, n_modality = modality_rank.shape
                elif self.more_config['modality_ranker'] == 'cosine_similarity':
                    image_embeds, depth_embeds, pc_embeds, norm_embeds, _ = x.split(token_length_list, dim=1)
                    modality_weight = self.prefusion([image_embeds, depth_embeds, pc_embeds, norm_embeds])
                    if self.more_config['discrete_token_number']:
                        modality_rank = get_rank_order(modality_weight)
                    else:
                        modality_rank = token_budgets_from_weight(modality_weight, self.total_multimodal_budgets)
                    # modality_rank = get_rank_order(modality_weight)
                    n_batch, n_modality = modality_rank.shape
                elif self.more_config['modality_ranker'] == 'random':
                    image_embeds, depth_embeds, pc_embeds, norm_embeds, _ = x.split(token_length_list, dim=1)
                    # print(x.shape)
                    modality_weight = self.prefusion([image_embeds, depth_embeds, pc_embeds, norm_embeds])
                    # modality_rank = get_rank_order(modality_weight)
                    
                    if self.more_config['discrete_token_number']:
                        modality_rank = get_rank_order(modality_weight)
                    else:
                        modality_rank = token_budgets_from_weight(modality_weight, self.total_multimodal_budgets)
                image_embeds, depth_embeds, pc_embeds, norm_embeds, _ = x.split(token_length_list, dim=1)
                # print(torch.all(x.isnan()))
                # print(x.shape, prefusion_attn.shape)
                # exit()
                llm_input_image_embeds = []
                random_rank = [1,2,3,0]
                # print(modality_rank)
                
                # print(image_embeds.shape, depth_embeds.shape, flow_embeds.shape, norm_embeds.shape, modality_rank)
                # print(x.shape, non_fused_inputs_embeds.shape)
                # exit()
                # print(modality_rank)
                for bi in range(n_batch):
                    if modality_rank is None:
                        random.shuffle(random_rank)
                        image_rank, depth_rank, pc_rank, norm_rank = random_rank
                    else:
                        image_rank, depth_rank, pc_rank, norm_rank = modality_rank[bi]
            
                    compressed_image_embeds = self.m_qformer['image'](image_embeds[bi:bi+1], image_rank)
                    compressed_depth_embeds = self.m_qformer['depth'](depth_embeds[bi:bi+1], depth_rank)
                    compressed_pc_embeds = self.m_qformer['pc'](pc_embeds[bi:bi+1], pc_rank)
                    compressed_norm_embeds = self.m_qformer['norm'](norm_embeds[bi:bi+1], norm_rank)
                    llm_input_image_embeds.append(
                        torch.cat([compressed_image_embeds, compressed_depth_embeds, compressed_pc_embeds, compressed_norm_embeds], dim=1)
                    )
                llm_input_image_embeds = torch.cat(llm_input_image_embeds, dim=0)
                # fusion_text_features = x[:, -1 *input_ids.size(1):,:]
                # print(llm_input_image_embeds.shape)
                
                # fusion_text_features=fusion_text_features*(~padding_mask).unsqueeze(-1).int()+inputs_embeds*padding_mask.unsqueeze(-1)


                image_embeds = llm_input_image_embeds.reshape(-1, llm_input_image_embeds.shape[-1])
                n_image_features = image_embeds.shape[0]
                inputs_embeds = non_fused_inputs_embeds

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)
                # print(image_mask.shape)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                # print(image_mask.shape, image_embeds.shape, input_ids.shape, inputs_embeds.shape, mask.sum())
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)
        # print(attention_mask.shape, input_ids.shape)
        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # print(position_i ds.shape, attention_mask.shape, inputs_embeds.shape)
        # exit()
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

def assign_qformer(model: Qwen2_5_VLForConditionalGenerationMore, more_config):
    model_dict = nn.ModuleDict()
    for mm in more_config['modality_list']:
        if more_config['token_pruning_method'] == 'qformer':
            model_dict[mm] = MultiLevelQFormerEncoder(
                model.config.hidden_size, more_config[mm], len(more_config['modality_list']), model.config.hidden_size,
                more_config=more_config
            )
        elif more_config['token_pruning_method'] == 'linear_pooling':
            # print("Use Multi-Level MLP", '!'*40)
            if more_config['parameterized_pooling']:
                n_original_token = more_config['n_original_tokens'] if more_config['n_original_tokens'] is not None else 10 * 256
                if mm == 'pc':
                    n_original_token = 5000
                model_dict[mm] = MultiLevelCompression(
                    model.config.hidden_size, more_config[mm], len(more_config['modality_list']), model.config.hidden_size,
                    more_config=more_config, origin_tokens = n_original_token,
                )
            else:
                model_dict[mm] = LinearPoolingParameterFree(
                    model.config.hidden_size, more_config[mm], len(more_config['modality_list']), model.config.hidden_size,
                    more_config=more_config
                )
        elif more_config['token_pruning_method'] == 'cosine_similarity':
            model_dict[mm] = CosineSimilarityPruning(
                    model.config.hidden_size, more_config[mm], len(more_config['modality_list']), model.config.hidden_size,
                    more_config=more_config
                )
        else:
            model_dict[mm] = QFormerEncoder(
                model.config.hidden_size, more_config[mm], model.config.hidden_size
            )
    model.register_module("m_qformer", model_dict)
    
    
def assign_prefusion(model: Qwen2_5_VLForConditionalGenerationMore, more_config):
    attn_type = model.config._attn_implementation
    n_prefusion_layers = more_config['n_prefusion_layers']
    model.config._attn_implementation = 'eager'
    if more_config['modality_ranker'] == 'attention':
        if not more_config['only_self_attention']:
            prefusion_layers=nn.ModuleList([Qwen2_5_VLDecoderLayer(model.config,layer_idx=i) for i in range(n_prefusion_layers)])
            # model.config._attn_implementation = attn_type
            # model.register_module("prefusion", prefusion_layers)
            # intialize from model
            # if not more_config['learnable_attention']:
            with torch.no_grad():
                for i in range(n_prefusion_layers):
                    prefusion_layers[i].load_state_dict(model.model.layers[i].state_dict())
        else:
            prefusion_layers=Attention(model.config.hidden_size)
            # model.register_module("prefusion", prefusion_layers)
    elif more_config['modality_ranker'] == 'random':
        prefusion_layers=RandomOrder()
        # model.register_module("prefusion", prefusion_layers)
    elif more_config['modality_ranker'] == 'cosine_similarity':
        prefusion_layers=CosineSimilarityRedundancy()
        # model.register_module("prefusion", prefusion_layers)
    # print(prefusion_layers)
    model.register_module("prefusion", prefusion_layers)
    model.config._attn_implementation = attn_type
    
class Qwen2_5_VLForConditionalGenerationMoreGRPO(Qwen2_5_VLForConditionalGenerationMore):
    def __init__(self, config, eval_model=False,
                 n_image=4, n_depth=4, n_norm=4, n_flow=4, multilevel_qformer=True,
                 n_prefusion_layers=1):
        super(Qwen2_5_VLForConditionalGenerationMoreGRPO).__init__(config, eval_model, n_image, n_depth, n_norm, n_flow, multilevel_qformer, n_prefusion_layers)
        self.config._attn_implementation = 'flash_attention_2'
        self._use_flash_attention_2 = True
        
