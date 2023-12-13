from icvrc.model_map import get_model_type_class
from icvrc.utils import sylabelize, normalize
from PIL import Image
from typing import List
import torch
import logging
import json
import os

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)


class VRC:
    """Tensorflow End-to-end Neural Response Generator Class."""

    def __init__(
            self,
            pretrained_model_path,
            question_max_len=48,
            answer_max_len=64,
            device=DEFAULT_DEVICE,
    ):
        self.torch_dtype = torch.bfloat16
        self.device = device
        self.model_type = self._get_model_type(pretrained_model_path)
        self._load_model(self.model_type, pretrained_model_path)
        self.question_max_len = question_max_len
        self.answer_max_len = answer_max_len

    @staticmethod
    def _get_model_type(pretrained_model_path):
        config_path = os.path.join(pretrained_model_path, "config.json")
        if not os.path.isfile(config_path):
            assert False, f"Không tìm thấy file config.json trong thư mục {pretrained_model_path}"
        with open(config_path, "r") as file:
            config = json.load(file)
        model_type = config.get("model_version", None)
        return model_type

    def _load_model(self, model_type, pretrained_model_path):
        """Load pretrained model and config."""
        classes = get_model_type_class(model_type)
        self.processor = classes.processor_class.from_pretrained(pretrained_model_path)
        self.model = classes.model_class.from_pretrained(
            pretrained_model_path, device_map=self.device, torch_dtype=self.torch_dtype
        )
        self.model.eval()

    def batch_infer(
        self,
        data: List[dict],
        num_beams: int = 4
    ) -> List[str]:
        """Sinh câu trả lời cho câu hỏi dựa theo thông tin trong hình ảnh.

        Args:
            data (List[dict]): Danh sách các cặp (image, question), mỗi cặp là 1 dict:
                {
                    "image_file": Đường dẫn đến file hình ảnh jpg,
                    "question": Câu hỏi tương ứng,
                }

        Returns:
            List[str]: Danh sách câu trả lời tương ứng
        """
        images = [
            Image.open(x["image_file"]).convert('RGB')
            for x in data
        ]
        texts = [self.preprocess_input(x["question"]) for x in data]
        inputs = self.processor(
            images,
            texts,
            truncation=True,
            max_length=self.question_max_len,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.torch_dtype)

        if any(["image_text" in item for item in data]):
            image_texts = [item.get("image_text", "") for item in data]
            imgtext_inputs = self.processor.tokenizer(
                image_texts,
                truncation=True,
                max_length=self.answer_max_len,
                padding="max_length",
                return_tensors="pt"
            )
            inputs["imgtxt_input_ids"] = imgtext_inputs["input_ids"].to(self.device)
            inputs["imgtxt_attention_mask"] = imgtext_inputs["attention_mask"].to(self.device)

        outs = self.model.generate(**inputs, max_new_tokens=self.answer_max_len, num_beams=num_beams)
        answers = self.processor.batch_decode(outs, skip_special_tokens=True)
        answers = [self.postprocess_output(a) for a in answers]
        return answers

    @staticmethod
    def preprocess_input(text):
        return sylabelize(normalize(text)).lower()

    @staticmethod
    def postprocess_output(text):
        return sylabelize(normalize(text.strip("."))).lower()


def get_root_path():
    """Trả về đường dẫn đến thư mục gốc."""
    root_path = os.path.dirname(os.path.realpath(__file__))
    return root_path
