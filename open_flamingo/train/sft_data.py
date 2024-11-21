import random
import json
from torch.utils.data import Dataset
import transformers
import re
import base64
import torch.distributed as dist
import torch
import re
import os
import time
import sys
from PIL import Image, ImageFile
from dataclasses import dataclass
import torchvision
import copy
from einops import rearrange

sys.path.append('path_to_trl')

from trl.trainer.utils import DPODataCollatorWithPadding
from typing import Any, Dict, Optional, Sequence, List

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_data(data_path):
    if "jsonl" in data_path:
        data_list = []
        with open(data_path, 'r') as file:
            for line in file:
                data_list.append(json.loads(line))
    else:
        with open(data_path, 'r') as file:
            data_list = json.load(file)
    return data_list

def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)

# def preprocess(questions, answers, tokenizer, has_image, max_len=1024) -> Dict:

#     # Apply prompt templates
#     input_ids, targets = [], []
#     for i, (question, answer) in enumerate(zip(questions, answers)):

#         input_id, target = [], []
#         tokenizer.padding_side = "right"
#         text_tensor1 = tokenizer(
#             question,
#             max_length=max_len,
#             truncation=True,
#             padding="max_length",
#             # return_tensors="pt",
#         )
#         _input_id = text_tensor1["input_ids"]
#         _target = [-100] * len(_input_id)
#         input_id += _input_id
#         target += _target
#         text_tensor2 = tokenizer(
#             answer,
#             max_length=max_len,
#             truncation=True,
#             padding="max_length",
#             # return_tensors="pt",
#         )
#         _input_id = text_tensor2["input_ids"]
#         if answer.startswith('Short answer:'):
#             _target = len(tokenizer('Short answer:').input_ids) * [-100] + _input_id[len(tokenizer('Short answer:').input_ids) :] 
#         elif answer.startswith('Answer:'):
#             _target = len(tokenizer('Answer:').input_ids) * [-100] + _input_id[len(tokenizer('Answer:').input_ids) :] 
#         else:
#             _target = _input_id
#             # print(answer)
#             # raise NotImplementedError
#         input_id += _input_id
#         target += _target

#         assert len(input_id) == len(target)
#         # input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
#         # target += [IGNORE_INDEX] * (max_len - len(target))
#         input_ids.append(input_id)
#         targets.append(target)
#     input_ids = torch.tensor(input_ids, dtype=torch.long)
#     targets = torch.tensor(targets, dtype=torch.long)

#     return dict(
#         input_ids=input_ids,  # tensor(bs x seq_len)
#         labels=targets,  # tensor(bs x seq_len)
#         # attention_mask=input_ids.ne(tokenizer.pad_token_id), # tensor(bs x seq_len)
#     )

def preprocess(sources, tokenizer, has_image, max_len=1024) -> Dict:

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        input_id, target = [], []
        tokenizer.padding_side = "right"
        if "prompt" in source:
            prompt = source["prompt"]
            # prompt = prompt.replace("<image>", "").strip()
            # prompt = "<image>" + prompt
            if "context" in source:
                prompt = source["context"] + prompt
            source["prompt"] = prompt
        else:
            prompt = None
        question = source['prompt']
        answer = source['answer'] + tokenizer.eos_token
        text_tensor1 = tokenizer(
            question,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            # return_tensors="pt",
        )
        _input_id = text_tensor1["input_ids"]
        _target = [-100] * len(_input_id)
        input_id += _input_id
        target += _target
        text_tensor2 = tokenizer(
            answer,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            # return_tensors="pt",
        )
        _input_id = text_tensor2["input_ids"]
        if answer.startswith('Short answer:'):
            _target = len(tokenizer('Short answer:').input_ids) * [-100] + _input_id[len(tokenizer('Short answer:').input_ids) :] 
        elif answer.startswith('Answer:'):
            _target = len(tokenizer('Answer:').input_ids) * [-100] + _input_id[len(tokenizer('Answer:').input_ids) :] 
        elif answer.startswith('Output:'):
            _target = len(tokenizer('Output:').input_ids) * [-100] + _input_id[len(tokenizer('Answer:').input_ids) :] 
        else:
            _target = _input_id
            # print(answer)
            # raise NotImplementedError
        input_id += _input_id
        target += _target

        assert len(input_id) == len(target)
        # input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        # target += [IGNORE_INDEX] * (max_len - len(target))
        input_ids.append(input_id)
        targets.append(target)

    
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
        # attention_mask=input_ids.ne(tokenizer.pad_token_id), # tensor(bs x seq_len)
    )


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path, tokenizer, image_folder, image_processor):
        super(LazySupervisedDataset, self).__init__()
        self.list_data_dict = []

        # Handle multiple JSON files specified in the data_path
        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}")
            dataset_paths = []
            for file_name in file_names:
                dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                rank0_print(f"Loading {full_path}")
                cur_data_dict = load_data(full_path)
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                self.list_data_dict.extend(cur_data_dict)
        else:
            dataset_paths = [data_path]
            rank0_print(f"Loading {data_path}")
            cur_data_dict = load_data(data_path)
            rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
            self.list_data_dict.extend(cur_data_dict)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.image_folder = image_folder
        self.image_processor = image_processor

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            # Calculate the length of the prompt, answer, chosen, and rejected text
            cur_len = len(sample["prompt"].split()) + len(sample["answer"].split())
            image_num = sample["prompt"].count("<image>")
            for context_item in sample["context"]:
                cur_len += len(context_item["value"].split())
                image_num += context_item["value"].count("<image>")
            # Add additional tokens if an image is present
            img_tokens = 128 * image_num if "image" in sample else 0
            length_list.append(cur_len + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            # Calculate the length of the prompt, answer, chosen, and rejected text
            cur_len = len(sample["prompt"].split()) + len(sample["answer"].split())
            for context_item in sample["context"]:
                cur_len += len(context_item["value"].split())
            # If the sample includes a video, the length is positive; otherwise, it is negative
            cur_len = cur_len if ("video" in sample or "image" in sample) else -cur_len
            length_list.append(cur_len)
        return length_list
    

    def process_image(self, image_files):
        image_folder = self.image_folder
        processor = self.image_processor
        # print(f"\n\nInspecting the image path, folder = {image_folder}, image={image_file}\n\n")
        images = []
        for image_file in image_files:
            try:
                image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
                images.append(image)
            except Exception as exn:
                print(f"Failed to open image {image_file}. Exception:", exn)
                raise exn
        
        image = [self.image_processor(s).unsqueeze(0) for s in images]
        image = torch.cat(image, dim=0)
        image = torchvision.transforms.RandomHorizontalFlip(p=0.5)(image)
        return image

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                pass

        assert False, "Failed to fetch sample."

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            if type(image_file) is list:
                image = self.process_image(image_file)
            else:
                image = self.process_image([image_file])
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        has_image = ("image" in self.list_data_dict[i]) or ("video" in self.list_data_dict[i])
        # data_dict = copy.deepcopy(self.list_data_dict[i])

        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])


        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        
        # prompt exist in the data
        data_dict["has_image"] = has_image
        return data_dict




@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        # if self.tokenizer.padding_side == "left":
        #     input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        # if self.tokenizer.padding_side == "left":
        #     input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # input_ids, labels, ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "id"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=-100)
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
        # batch = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id), ids=ids)

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            images = torch.stack(images, dim=0)
            images = rearrange(images, "b (t f) c h w -> b t f c h w", f=1)

            batch["images"] = images

        # print(batch.keys())
        # print(padded_batch["input_ids"].shape)
        # print(padded_batch["labels"].shape)
        # print(padded_batch["attention_mask"].shape)
        # import time
        # time.sleep(1000)
        return batch


def make_supervised_data_module(tokenizer, data_path, image_folder, image_processor) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(data_path=data_path, tokenizer=tokenizer, image_folder=image_folder, image_processor=image_processor)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
