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

def preprocess(questions, answers, tokenizer, has_image, max_len=1024) -> Dict:

    # Apply prompt templates
    input_ids, targets = [], []
    for i, (question, answer) in enumerate(zip(questions, answers)):

        input_id, target = [], []
        tokenizer.padding_side = "right"
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


class DPODataset(Dataset):
    """Dataset for DPODataset fine-tuning."""

    def __init__(self, data_path, tokenizer, image_folder, image_processor):
        super(DPODataset, self).__init__()
        # Handle multiple JSON files specified in the data_path
        self.list_data_dict = []

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
            cur_len = len(sample["prompt"].split()) + len(sample["answer"].split()) + len(sample["chosen"].split()) + len(sample["rejected"].split())
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
            cur_len = len(sample["prompt"].split()) + len(sample["answer"].split()) + len(sample["chosen"].split()) + len(sample["rejected"].split())
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

        # still fail, most likely to be path issue or cloud disk issue, retry the same sample for longer
        # for attempt_idx in range(num_final_retries):
        #     try:
        #         sample = self._get_item(i)
        #         return sample
        #     except Exception as e:
        #         # sleep 1s in case it is a cloud disk issue
        #         print(f"[Final try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
        #         time.sleep(1)

        # Finally raise exception on failing.
        assert False, "Failed to fetch sample."

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        suffix = None
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            if type(image_file) is list:
                image = self.process_image(image_file)
            else:
                image = self.process_image([image_file])
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        has_image = ("image" in self.list_data_dict[i]) or ("video" in self.list_data_dict[i])
        data_dict = copy.deepcopy(self.list_data_dict[i])  # inplace modification following

        if "prompt" in data_dict:
            prompt = data_dict["prompt"]
            # prompt = prompt.replace("<image>", "").strip()
            # prompt = "<image>" + prompt
            if "context" in data_dict:
                prompt = data_dict["context"] + prompt
            data_dict["prompt"] = prompt
        else:
            prompt = None
        

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        
        # prompt exist in the data
        data_dict["has_image"] = has_image
        return data_dict


@dataclass
class DPODataCollator(DPODataCollatorWithPadding):
    """Collate examples for DPO fine-tuning."""

    # tokenizer: transformers.PreTrainedTokenizer

    def collate(self, batch):
        # first, pad everything to the same length
        # input_ids, labels = tuple([instance[key] for instance in instances]
        #                           for key in ("input_ids", "labels"))
        # input_ids = torch.nn.utils.rnn.pad_sequence(
        #     input_ids,
        #     batch_first=True,
        #     padding_value=self.tokenizer.pad_token_id)
        # labels = torch.nn.utils.rnn.pad_sequence(labels,
        #                                          batch_first=True,
        #                                          padding_value=IGNORE_INDEX)
        # input_ids = input_ids[:, :self.tokenizer.model_max_length]
        # labels = labels[:, :self.tokenizer.model_max_length]
        # batch = dict(
        #     input_ids=input_ids,
        #     labels=labels,
        #     attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        # )
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                # if "prompt" in k:
                #     to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                # else:
                to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith("_input_ids"):
                    padding_value = self.tokenizer.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = self.label_pad_token_id
                else:
                    continue
                # elif k.endswith("_attention_mask"):
                #     padding_value = self.padding_value
                # else:
                #     raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = torch.nn.utils.rnn.pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                # for the prompt, flip back so padding is on left side
                # if "prompt" in k:
                #     padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]
        for k in ["chosen_input_ids", "rejected_input_ids"]:
            attn_k = k.replace("input_ids", "attention_mask")
            padded_batch[attn_k] = padded_batch[k].ne(self.tokenizer.pad_token_id)
        return padded_batch

    def tokenize_batch_element(self, prompt: str, chosen: str, rejected: str, has_image: bool = True) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        # import pdb; pdb.set_trace()
        batch = {}

        # chosen_sources = prompt + chosen + self.tokenizer.eos_token
        # rejected_sources = prompt + rejected + self.tokenizer.eos_token
        chosen_data_dict = preprocess([prompt], [chosen + self.tokenizer.eos_token], self.tokenizer, has_image=has_image)
        # chosen_data_dict['attention_mask'] = chosen_data_dict["input_ids"].ne(self.tokenizer.pad_token_id)

        rejected_data_dict = preprocess([prompt], [rejected + self.tokenizer.eos_token], self.tokenizer, has_image=has_image)
        # rejected_data_dict['attention_mask'] = rejected_data_dict["input_ids"].ne(self.tokenizer.pad_token_id)

        chosen_data_dict = {k: v[0] for k, v in chosen_data_dict.items()}
        rejected_data_dict = {k: v[0] for k, v in rejected_data_dict.items()}

        for k, toks in {
            "chosen": chosen_data_dict,
            "rejected": rejected_data_dict,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens
        return batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        tokenized_batch = []
        Xs, keys = [], []
        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["chosen"]
            rejected = feature["rejected"]
            has_image = feature["has_image"]
            # Xs.append(feature[has_X])
            # keys.append(has_X)

            batch_element = self.tokenize_batch_element(prompt, chosen, rejected, has_image=has_image)
            tokenized_batch.append(batch_element)

        # return collated batch
        padded_batch = self.collate(tokenized_batch)
        # import pdb;pdb.set_trace()
        if "image" in features[0]:
            # instances[1]['image'][0][0].shape
            # torch.Size([5, 3, 224, 224])
            images = [instance["image"] for instance in features]
            images = torch.stack(images, dim=0)
            images = rearrange(images, "b (t f) c h w -> b t f c h w", f=1)

            # padded_batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            # padded_batch["modalities"] = [im[2] for im_list in images for im in im_list]
            # images = [im for im_list in images for im in im_list]
            # import pdb;pdb.set_trace()

            padded_batch["images"] = images
            # padded_batch["images"] =[padded_batch["modalities"], images]

        return padded_batch


def make_dpo_data_module(tokenizer, data_path, image_folder, image_processor) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = DPODataset(data_path=data_path, tokenizer=tokenizer, image_folder=image_folder, image_processor=image_processor)
    return train_dataset
