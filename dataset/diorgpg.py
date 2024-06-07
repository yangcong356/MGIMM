from typing import Dict, Sequence
import json
import torch
import transformers
import copy
import os
from dataclasses import dataclass
import random

from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from utils.utils import rank0_print
from utils.constants import IGNORE_INDEX
from .data_modules import preprocess, preprocess_multimodal, apply_boxes
import pdb


class DIORGPGDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, annotations_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(DIORGPGDataset, self).__init__()
        data_dict = json.load(open(annotations_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode", local_rank=data_args.local_rank)
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.annotations = data_dict

    def __len__(self):
        return len(self.annotations)

    @property
    def lengths(self):
        length_list = []
        for sample in self.annotations:
            img_tokens = 128 if 'image_id' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.annotations:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image_id' in sample else -cur_len
            length_list.append(cur_len)
        return length_list


    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        ann = self.annotations[index]
        if isinstance(index, int):
            sources = [ann]
        
        # pdb.set_trace()

        if 'image_id' in ann:
            image_file_name = ann['file_name']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file_name)).convert('RGB')
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image_id' in ann))

        if isinstance(index, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])

        # image exist in the data
        if 'image_id' in ann:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        
        temp_bbox = []
        if 'bbox' in ann:
            for i in range(0,len(ann['bbox'])):
                x1, y1, w, h = ann['bbox'][i]
                bbox = np.array([x1, y1, x1 + w, y1 + h], dtype=np.float32)
                input_img_size = (ann['width'], ann['height'])
                target_length = self.data_args.input_image_size
                bbox = apply_boxes(bbox, input_img_size, target_length)
                bbox_torch = torch.as_tensor(bbox, dtype=torch.float)
                temp_bbox.append(bbox_torch)
            data_dict['bbox'] = torch.stack(temp_bbox,dim=1)
        else:
            bbox = np.array([100.0, 100.0, 100.0 + 100.0, 100.0 + 100.0], dtype=np.float32)
            bbox_torch = torch.as_tensor(bbox, dtype=torch.float)
            data_dict['bbox'] = bbox_torch

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        )

        
        if 'image' in instances[0]:
            bbox = [instance['bbox'] for instance in instances]
            images = [instance['image'] for instance in instances]
            # print(bbox)
            # print(bbox[0])
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
                batch['bbox'] = bbox
                # print(batch['bbox'])
            else:
                batch['images'] = images
                batch['bbox'] = bbox
        

        return batch
    


def make_supervised_data_module(tokenizer,
                                data_args) :
    """Make dataset and collator for supervised fine-tuning."""

    if data_args.dataset_config is not None:
        dataset_config = OmegaConf.load(data_args.dataset_config)
    
    data_args.image_folder = dataset_config['datasets'].pop("images")
    dataset_annotations =  dataset_config['datasets'].pop('annotations')

    datasets = []
    # datasets = dict()
    for split in dataset_annotations.keys():
        # pdb.set_trace()
        data_args.dataset_annotations = dataset_annotations[split]
        temp_data = build_dataset(dataset_config['datasets'],
                                    tokenizer=tokenizer,
                                    data_args=data_args)
        # pdb.set_trace()
        # temp_data.__getitem__(0)
        datasets.append(temp_data)

    # pdb.set_trace()
    datasets_concat = ConcatDataset(datasets)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(train_dataset=datasets_concat,
                eval_dataset=None,
                data_collator=data_collator)

def build_dataset(dataset_config,
                  tokenizer: transformers.PreTrainedTokenizer,
                  data_args):

    dataset_name = dataset_config['dataset_name']
    if dataset_name == 'diorgpg':
        dataset = DIORGPGDataset(
            data_args.dataset_annotations,
            tokenizer=tokenizer,
            data_args=data_args
        )
    else:
        raise NotImplementedError

    return dataset



class ConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)

    def collater(self, samples):
        # TODO For now only supports datasets with same underlying collater implementations

        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)