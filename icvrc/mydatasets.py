import os
import json
from icvrc.utils import normalize
import torch
from PIL import Image
from datasets import Dataset


class VRCTorchDatasetV1():
    raw_features = ["image_file", "question", "answer"]

    def __init__(
        self,
        data_dir,
        processor,
        question_max_len=48,
        answer_max_len=64,
        pad_label_id=-100,
        max_data_number=None,
        images_folder="training-images",
        data_file="vlsp2023_train_data.json",
    ):
        self.processor = processor
        self.question_max_len = question_max_len
        self.answer_max_len = answer_max_len
        self.pad_label_id = pad_label_id

        self._data = self._read_data(
            os.path.join(data_dir, images_folder),
            data_file=os.path.join(data_dir, data_file)
        )
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
            remove_columns=self.raw_features
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


class VRCTorchDatasetV2(VRCTorchDatasetV1):
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
        imgtext_encoding = self.processor.tokenizer(
            examples["image_text"],
            max_length=self.answer_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoding["imgtxt_input_ids"] = imgtext_encoding["input_ids"]
        encoding["imgtxt_attention_mask"] = imgtext_encoding["attention_mask"]

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
