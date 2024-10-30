import os
import json
from icvrc.utils import normalize
import torch
from PIL import Image
from datasets import Dataset


class VRCTorchDatasetForMBLIP():
    raw_features = ["image_file", "question", "answer"]

    def __init__(
        self,
        processor,
        question_max_len=48,
        answer_max_len=64,
        pad_label_id=-100,
        max_data_number=None,
        images_folder="training-images",
        data_file="vlsp2023_train_data.json",
        is_decoder_only_lm=False,
    ):
        self.processor = processor
        self.question_max_len = question_max_len
        self.answer_max_len = answer_max_len
        self.pad_label_id = pad_label_id
        self.is_decoder_only_lm = is_decoder_only_lm

        self._data = self._read_data(images_folder, data_file)
        if max_data_number is not None:
            self._data = self._data[:max_data_number]

    def _read_data(self, image_folder, data_file):
        with open(data_file, "r") as file:
            raw_data = json.load(file)

        data = []
        for _, item in raw_data["annotations"].items():
            image_file = os.path.join(image_folder, raw_data["images"][str(item["image_id"])])
            data.append({
                "image_file": image_file,
                "question": normalize(item["question"]),
                "answer": normalize(item["answer"]),
            })
        return data

    def __len__(self):
        return len(self._data)

    def _generator(self):
        for item in self._data:
            yield item

    def get_hf_dataset(self):
        hf_train_dataset = Dataset.from_generator(self._generator)
        hf_train_dataset = hf_train_dataset.map(
            self._preprocess_data,
            batched=True,
            remove_columns=self.raw_features,
            num_proc=4,
            keep_in_memory=True,
        )
        return hf_train_dataset

    def _preprocess_data(self, examples):
        images = [
            Image.open(image_file).convert('RGB')
            for image_file in examples["image_file"]
        ]
        encoding = self.processor(
            images,
            examples["question"],
            max_length=self.question_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        if self.is_decoder_only_lm:
            bos_token = self.processor.tokenizer.bos_token
            eos_token = self.processor.tokenizer.eos_token
            answers = [f"Answer: {bos_token}{a}{eos_token}" for a in examples["answer"]]
            answers_encoding = self.processor.tokenizer(
                answers,
                max_length=self.answer_max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                padding_side="right",
            )
            encoding["input_ids"] = torch.cat((encoding["input_ids"], answers_encoding["input_ids"]), dim=1)
            encoding["attention_mask"] = torch.cat((encoding["attention_mask"], answers_encoding["attention_mask"]),
                                                   dim=1)
        else:
            answers_encoding = self.processor.tokenizer(
                examples["answer"],
                max_length=self.answer_max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        label_ids = answers_encoding["input_ids"]
        label_ids = torch.where(label_ids != self.processor.tokenizer.pad_token_id, label_ids, self.pad_label_id)
        encoding["labels"] = label_ids

        return encoding


class VRCTorchDatasetForMBLIPwST(VRCTorchDatasetForMBLIP):
    raw_features = ["image_file", "image_text", "question", "answer"]

    def _read_data(self, image_folder, data_file):
        with open(data_file, "r") as file:
            raw_data = json.load(file)

        data = []
        for _, item in raw_data["annotations"].items():
            image_file = os.path.join(image_folder, raw_data["images"][str(item["image_id"])])
            image_text = item["image_text"]
            data.append({
                "image_file": image_file,
                "image_text": normalize(image_text).lower(),
                "question": normalize(item["question"]),
                "answer": normalize(item["answer"]),
            })
        return data

    def _preprocess_data(self, examples):
        images = [
            Image.open(image_file).convert('RGB')
            for image_file in examples["image_file"]
        ]
        encoding = self.processor(
            images,
            examples["question"],
            max_length=self.question_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        if self.is_decoder_only_lm:
            bos_token = self.processor.tokenizer.bos_token
            eos_token = self.processor.tokenizer.eos_token
            answers = [f"Answer: {bos_token}{a}{eos_token}" for a in examples["answer"]]
            answers_encoding = self.processor.tokenizer(
                answers,
                max_length=self.answer_max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                padding_side="right",
            )
            encoding["input_ids"] = torch.cat((encoding["input_ids"], answers_encoding["input_ids"]), dim=1)
            encoding["attention_mask"] = torch.cat((encoding["attention_mask"], answers_encoding["attention_mask"]),
                                                   dim=1)
        else:
            answers_encoding = self.processor.tokenizer(
                examples["answer"],
                max_length=self.answer_max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        label_ids = answers_encoding["input_ids"]
        label_ids = torch.where(label_ids != self.processor.tokenizer.pad_token_id, label_ids, self.pad_label_id)
        encoding["labels"] = label_ids

        imgtext_encoding = self.processor.tokenizer(
            examples["image_text"],
            max_length=self.answer_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoding["imgtxt_input_ids"] = imgtext_encoding["input_ids"]
        encoding["imgtxt_attention_mask"] = imgtext_encoding["attention_mask"]
        return encoding
