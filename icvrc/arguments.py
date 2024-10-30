"""Arguments for training."""
from dataclasses import dataclass, field
from typing import Optional
from icvrc.model_map import MODEL_TYPES
from transformers.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch."""

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: str = field(
        default="mblip-mt0",
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    freeze_vision_model: bool = field(
        default=False,
        metadata={
            "help": "Đóng băng trọng số Vision Model"
        },
    )
    freeze_language_model: bool = field(
        default=False,
        metadata={
            "help": "Đóng băng trọng số Language Model"
        },
    )


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    train_image_folder: str = field(
        metadata={"help": "Thư mục chứa hình ảnh cho dữ liệu train"},
    )
    train_data_file: str = field(
        metadata={"help": "Thư mục chứa dữ liệu train"},
    )
    dev_image_folder: str = field(
        default=None, metadata={"help": "Thư mục chứa hình ảnh cho dữ liệu dev"},
    )
    dev_data_file: str = field(
        default=None, metadata={"help": "Thư mục chứa dữ liệu dev"},
    )
    question_max_len: int = field(
        default=64, metadata={"help": "Chiều dài tối đa của câu hỏi, tính theo số token."},
    )
    answer_max_len: int = field(
        default=64, metadata={"help": "Chiều dài tối đa của câu trả lời, tính theo số token."},
    )
    max_data_number: Optional[int] = field(
        default=None,
        metadata={"help": "Số lượng dữ liệu tối đa cho mỗi dataset, sử dụng khi muốn test code với số lượng nhỏ dữ liệu"},
    )
