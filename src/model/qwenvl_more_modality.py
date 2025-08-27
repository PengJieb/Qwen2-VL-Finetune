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
from torch.nn import CrossEntropyLoss
import torch.nn as nn

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, _CONFIG_FOR_DOC, Qwen2_5_VLDecoderLayer
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import QWEN2_5_VL_INPUTS_DOCSTRING, Qwen2_5_VLCausalLMOutputWithPast
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor, Qwen2_5_VLProcessorKwargs
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.image_utils import ImageInput, VideoInput
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs


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
        return x

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
        x = query + self.attn(self.norm1(query), x)
        x = x + self.mlp(self.norm2(x))
        return x

class MultiLevelQFormerEncoder(nn.Module):
    def __init__(self, feature_size, num_patches, n_modality, embed_dim, attn_drop=0.25, drop = 0.25, num_heads=4, qkv_bias=False, mlp_ratio = 4, act_layer=nn.GELU):
        super().__init__()
        n_unite = num_patches // n_modality
        self.n_patch_list = [n_unite + 2 * i * n_unite for i in range(n_modality)]
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

    def forward(self, x:torch.Tensor, rank):

        x = self.context_proj(self.context_norm(x))
        
        B, C, D = x.shape
        
        query = self.query_tokens[:self.n_patch_list[rank]].unsqueeze(0).repeat(B, 1, 1)
        x = query + self.attn(self.norm1(query), x)
        x = x + self.mlp(self.norm2(x))
        return x

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

class Qwen2_5_VLForConditionalGenerationMore(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config, eval_model=False,
                 n_image=4, n_depth=4, n_norm=4, n_flow=4, multilevel_qformer=True,
                 n_prefusion_layers=1):
        super().__init__(config)
        self.config = config
        # self.n_per_img = n_per_img
        # self.n_prefusion_layers = n_prefusion_layers
        # self.multilevel_qformer = config.multilevel_qformer
        # if not lazy_load:
        #     assign_qformer(self, modalities, multilevel_qformer)
        #     assign_prefusion(self, n_prefusion_layers)
        # print(n_image, n_depth, n_norm, n_flow, multilevel_qformer, n_prefusion_layers)
        if eval_model:
            assign_qformer(self, {"image": n_image, 'depth': n_depth, 'norm': n_norm, 'flow': n_flow},
                           multilevel_qformer=multilevel_qformer)
            assign_prefusion(self, n_prefusion_layers)
    
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
                # print(pixel_values.shape, depth_values.shape, flow_values.shape, norm_values.shape, image_grid_thw.shape)
                # print(image_grid_thw)
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
                
                # compressed_image_embeds = self.m_qformer['image'](image_embeds)
                # compressed_depth_embeds = self.m_qformer['depth'](depth_embeds)
                # compressed_flow_embeds = self.m_qformer['flow'](flow_embeds)
                # compressed_norm_embeds = self.m_qformer['norm'](norm_embeds)
                # print(attention_mask.shape)
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
                # print(padding_mask.shape) 
                mask = torch.cat((torch.zeros((padding_mask.size(0),
                                               global_image_features_length),device=padding_mask.device).bool(), padding_mask),dim=1)
                # print(mask.shape, global_image_features_length)
                prefusion_position_ids = (~mask).int().long().cumsum(-1) - 1
                prefusion_position_ids.masked_fill_((~mask).int() == 0, 1)
                
                
                # if getattr(self, "_use_flash_attention_2", False) or getattr(self.config, "_attn_implementation", "") == "flash_attention_2":
                #     prefusion_attention_mask =(~mask).int()
                # else: 
                # prefusion_attention_mask =_prepare_4d_causal_attention_mask(~mask, (x.size(0), x.size(1)), x, 0)
                # Expanding 4d mask
                seq_len = mask.size(1)
                prefusion_attention_mask =(~mask).int()
                prefusion_attention_mask = prefusion_attention_mask.unsqueeze(1)
                prefusion_attention_mask = prefusion_attention_mask.unsqueeze(3)
                prefusion_attention_mask = prefusion_attention_mask.expand(-1, -1, -1, seq_len)

                if prefusion_position_ids.dim() == 2:
                    prefusion_position_ids = prefusion_position_ids[None, ...].expand(3, prefusion_position_ids.shape[0], -1)
                prefusion_position_embeddings = self.model.rotary_emb(x, prefusion_position_ids)
                if hasattr(self, 'prefusion'): 
                    for layer in self.prefusion:
                        lout = layer(x,
                                  attention_mask=prefusion_attention_mask, position_embeddings=prefusion_position_embeddings,
                                  output_attentions=True)
                        x = lout[0]
                        prefusion_attn = lout[1]
                        # print(prefusion_attn.shape)
                image_embeds, depth_embeds, flow_embeds, norm_embeds, _ = x.split(token_length_list, dim=1)
                prefusion_attn = prefusion_attn.sum(dim=-2).mean(dim=1) # batch size, seq_len
                image_weight, depth_weight, flow_weight, norm_weight, text_weight = prefusion_attn.split(token_length_list, dim=1)
                image_weight, depth_weight, flow_weight, norm_weight = image_weight.sum(-1, keepdim=True), depth_weight.sum(-1, keepdim=True), flow_weight.sum(-1, keepdim=True), norm_weight.sum(-1, keepdim=True)
                modality_weight = torch.cat([image_weight, depth_weight, flow_weight, norm_weight], dim=1) # batch, n_modality
                modality_rank = get_rank_order(modality_weight)
                n_batch, n_modality = modality_rank.shape
                llm_input_image_embeds = []
                for bi in range(n_batch):
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

                # fusion_text_features=fusion_text_features*(~padding_mask).unsqueeze(-1).int()+inputs_embeds*padding_mask.unsqueeze(-1)


                image_embeds = llm_input_image_embeds.reshape(-1, llm_input_image_embeds.shape[-1])
                n_image_features = image_embeds.shape[0]
                # print(n_image_features, n_image_tokens)
                # print(non_fused_inputs_embeds.shape, fusion_text_features.shape)
                inputs_embeds = non_fused_inputs_embeds
                # if n_image_tokens != n_image_features:
                #     raise ValueError(
                #         f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                #     )

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


def assign_qformer(model: Qwen2_5_VLForConditionalGenerationMore, modalities, multilevel_qformer = False):
    model_dict = nn.ModuleDict()
    for mm in modalities:
        if multilevel_qformer:
            model_dict[mm] = MultiLevelQFormerEncoder(
                model.config.hidden_size, modalities[mm], len(modalities), model.config.hidden_size
            )
        else:
            model_dict[mm] = QFormerEncoder(
                model.config.hidden_size, modalities[mm], model.config.hidden_size
            )
    model.register_module("m_qformer", model_dict)
    
    
def assign_prefusion(model: Qwen2_5_VLForConditionalGenerationMore, n_prefusion_layers=3):
    attn_type = model.config._attn_implementation
    model.config._attn_implementation = 'eager'
    
    prefusion_layers=nn.ModuleList([Qwen2_5_VLDecoderLayer(model.config,layer_idx=i) for i in range(n_prefusion_layers)])
    model.config._attn_implementation = attn_type
    model.register_module("prefusion", prefusion_layers)
    
    
class Qwen2_5_VLForConditionalGenerationMoreGRPO(Qwen2_5_VLForConditionalGenerationMore):
    def __init__(self, config, eval_model=False,
                 n_image=4, n_depth=4, n_norm=4, n_flow=4, multilevel_qformer=True,
                 n_prefusion_layers=1):
        super(Qwen2_5_VLForConditionalGenerationMoreGRPO).__init__(config, eval_model, n_image, n_depth, n_norm, n_flow, multilevel_qformer, n_prefusion_layers)
        self.config._attn_implementation = 'flash_attention_2'
        self._use_flash_attention_2 = True
        
