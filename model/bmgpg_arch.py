#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import math

from .RegionFeatureExtractor.prompt_encoder import PromptEncoder
from .RegionFeatureExtractor.region_interactive_layer import TwoWayTransformer
from .VisionEncoder.builder import build_vision_tower
from .MultiModalProjector.mmp_builder import build_vision_projector

import pdb

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class BMGPGMetaModel:

    def __init__(self, config):
        super(BMGPGMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=False)
            self.mm_projector = build_vision_projector(config)

        if hasattr(config, "bbox_interactive_encoder"):
            self.prompt_encoder = PromptEncoder(embed_dim=config.embed_dim, 
                                           image_embedding_size=config.image_embedding_size, 
                                           input_image_size=config.input_image_size)
            self.region_interactive_layer = TwoWayTransformer(depth=config.depth,
                                                         embedding_dim=config.embedding_dim,
                                                         mlp_dim=config.mlp_dim,
                                                         num_heads=config.num_heads)
            pass
        
    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        return vision_tower
    
    def get_prompt_encoder(self):
        prompt_encoder = getattr(self, 'prompt_encoder', None)
        region_interactive_layer = getattr(self, 'region_interactive_layer', None)
        return prompt_encoder, region_interactive_layer

    def initialize_vision_modules(self, model_args):
        vision_tower = model_args.vision_tower
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        # pdb.set_trace()

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            self.vision_tower = vision_tower
        else:
            vision_tower = self.vision_tower
            vision_tower.load_model()
        
        if model_args.stage =="stage2":
            pass
        else:
            self.config.bbox_interactive_encoder = model_args.region_feature_extractor
            if self.get_prompt_encoder()[0] is None:
                self.config.embed_dim = vision_tower.hidden_size
                self.config.image_embedding_size = (int(math.sqrt(vision_tower.num_patches)), int(math.sqrt(vision_tower.num_patches)))
                self.config.input_image_size = model_args.prompt_image_size
                self.config.depth = model_args.twtrans_depth
                self.config.embedding_dim = vision_tower.hidden_size
                self.config.mlp_dim = model_args.twtrans_mlp_dim
                self.config.num_heads = model_args.twtrans_num_heads
                prompt_encoder = PromptEncoder(embed_dim=self.config.embed_dim, 
                                            image_embedding_size=self.config.image_embedding_size, 
                                            input_image_size=self.config.input_image_size)
                region_interactive_layer = TwoWayTransformer(depth=self.config.depth,
                                                            embedding_dim=self.config.embedding_dim,
                                                            mlp_dim=self.config.mlp_dim,
                                                            num_heads=self.config.num_heads)
                self.prompt_encoder = prompt_encoder
                self.region_interactive_layer = region_interactive_layer
            else:
                prompt_encoder = self.prompt_encoder
                region_interactive_layer = self.region_interactive_layer

        # pdb.set_trace()
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        if model_args.stage == "stage2":
            self.config.mm_hidden_size = vision_tower.hidden_size
        else:
            self.config.mm_hidden_size = region_interactive_layer.get_embedding_dim
            
        self.config.mm_vision_select_layer = model_args.mm_vision_select_layer
        self.config.mm_vision_select_feature = model_args.mm_vision_select_feature
        self.config.stage = model_args.stage

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            if model_args.stage == "stage2":
                # In case it is frozen by LoRA
                for p in self.mm_projector.parameters():
                    p.requires_grad = True
            elif model_args.stage == "stage3":
                for p in self.mm_projector.parameters():
                    p.requires_grad = True
                for p in self.prompt_encoder.parameters():
                    p.requires_grad = True
                for p in self.region_interactive_layer.parameters():
                    p.requires_grad = True
            else:
                raise NotImplementedError

        if model_args.stage == "stage2":
            if pretrain_mm_mlp_adapter is not None:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
                self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
        elif model_args.stage == "stage3":
            if pretrain_mm_mlp_adapter is not None:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
                self.prompt_encoder.load_state_dict(get_w(mm_projector_weights, 'prompt_encoder'), strict=False)
                self.region_interactive_layer.load_state_dict(get_w(mm_projector_weights, 'region_interactive_layer'))
                self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
        


class BMGPGMetaForCausalLM(ABC):
    def __init__(self):
        super(BMGPGMetaForCausalLM, self).__init__()

    '''
        使用 @abstractmethod 装饰器，这个方法被声明为抽象方法。这意味着任何继承 LlavaMetaForCausalLM 的子类都必须实现 get_model 方法。
    '''
    @abstractmethod
    def get_model(self):
        pass

    '''
        这是一个普通方法，它调用了类中的 get_model 方法，并进一步调用 get_model 返回对象的 get_vision_tower 方法。
        这个方法的实现依赖于 get_model 方法的返回值。它假设 get_model 返回的对象具有 get_vision_tower 方法。
    '''
    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def get_prompt_encoder(self):
        return self.get_model().get_prompt_encoder()

    def encode_images(self, images, bbox):
        image_features = self.get_model().get_vision_tower()(images)    # [64,576,1024]
        # pdb.set_trace()
        
        if self.config.stage == "stage1" or self.config.stage == "stage3":
            prompt_encoder, region_interactive_layer = self.get_model().get_prompt_encoder()
            if type(bbox) == list:
                # if image_features.shape[0] == 1:
                signle_region_features = []
                # pdb.set_trace()
                sparse_prompt_embeddings = prompt_encoder(bbox[0].permute(1,0,2))   # [N_box,2,channel]
                for index in range(0, bbox[0].shape[1]): # [1, N_box, channel]
                    signle_region_feature, _ = region_interactive_layer(image_embedding=image_features,
                                                image_pe=prompt_encoder.get_dense_pe(),
                                                point_embedding=sparse_prompt_embeddings[index,:,:].unsqueeze(0))
                    signle_region_features.append(signle_region_feature)
                region_features = torch.cat(signle_region_features, dim=0).reshape(1,-1,image_features.shape[2])
                del signle_region_features, signle_region_feature
                # else:
                # batch_region_features = []
                # for i in range(0,image_features.shape[0]):
                #     signle_region_features = []
                #     for index in range(0, bbox[i].shape[1]):
                #         sparse_prompt_embeddings = prompt_encoder(bbox[i][:,index,:].unsqueeze(1))  # [N_box,2,channel]
                #         region_features, _ = region_interactive_layer(image_embedding=image_features[i,:,:].unsqueeze(0),
                #                                     image_pe=prompt_encoder.get_dense_pe(),
                #                                     point_embedding=sparse_prompt_embeddings)
                #         signle_region_features.append(region_features)
                #     signle_region_features = torch.mean(torch.cat(signle_region_features, dim=0), dim=0).unsqueeze(dim=0)
                #     batch_region_features.append(signle_region_features)
                # region_features = torch.cat(batch_region_features, dim=0)
                # del batch_region_features, signle_region_features
            else:
                sparse_prompt_embeddings = prompt_encoder(bbox)     # [B, 2, 1024]
                region_features, _ = region_interactive_layer(image_embedding=image_features,  #[B,2,1024], [B,576,1024]
                                                image_pe=prompt_encoder.get_dense_pe(),
                                                point_embedding=sparse_prompt_embeddings)
            self.get_model().mm_projector.to(device=region_features.device, dtype=region_features.dtype)
            image_features = self.get_model().mm_projector(image_features)
            region_features = self.get_model().mm_projector(region_features)
            return region_features, image_features
        else:
            image_features = self.get_model().mm_projector(image_features)
            return image_features


    def prepare_inputs_labels_for_multimodal(
        self, input_ids, bbox, attention_mask, past_key_values, labels, images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        # pdb.set_trace()

        if self.config.stage == "stage1":
            # pdb.set_trace()
            region_features, _ = self.encode_images(images, bbox)
        elif self.config.stage == "stage3":
            # pdb.set_trace()
            region_features, image_features = self.encode_images(images, bbox)
            # region_features = region_features.reshape(image_features.shape[0],-1,image_features.shape[2])
            global_region_features = torch.cat([image_features, region_features], dim=1)
        else:
            image_features = self.encode_images(images, bbox)


        
#############################################################################################
## 利用SAM Decoder中的Transformer结构交互bounding box和image embedding 构建一个可学习的token用于存储region-level features，用于引导因果语言建模
#############################################################################################


        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                if self.config.stage == "stage1":
                    cur_image_features = region_features[cur_image_idx]
                elif self.config.stage == "stage3":
                    cur_image_features = global_region_features[cur_image_idx]
                else:
                    cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                if self.config.stage == "stage1":
                    cur_image_features = region_features[cur_image_idx]
                elif self.config.stage == "stage3":
                    cur_image_features = global_region_features[cur_image_idx]
                else:
                    cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                        cur_labels = cur_labels[image_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        mask_tokens = ['<mask>', '<pos>']
        num_new_tokens = tokenizer.add_tokens(mask_tokens, special_tokens=True)

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
        
        for m in self.modules():
            m.tokenizer = tokenizer