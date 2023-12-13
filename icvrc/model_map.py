"""Configuration model mapping."""
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    # InstructBlipProcessor,
    # InstructBlipForConditionalGeneration
)
from icvrc.models.blip2_mt0 import Blip2ForT5VQA, Blip2ForT5VQAV2, Blip2ForT5VQAV3
from icvrc.mydatasets import VRCTorchDatasetV1, VRCTorchDatasetV2
import os


class ModelVersionMap():
    def __init__(
        self,
        processor_class,
        model_class,
        dataset_class,
        description=""
    ):
        self.processor_class = processor_class
        self.model_class = model_class
        self.dataset_class = dataset_class
        self.description = description

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        return self.__str__


DEFAULT_MODEL_TYPE = "mblip-mt0"
MAP_CONFIG_TOKENIZER_MODEL_CLASS = {
    "mblip-mt0": ModelVersionMap(
        processor_class=Blip2Processor,
        model_class=Blip2ForT5VQA,
        dataset_class=VRCTorchDatasetV1,
        description="Blip2 Model"
    ),
    "mblip-mt0-v2": ModelVersionMap(
        processor_class=Blip2Processor,
        model_class=Blip2ForT5VQAV2,
        dataset_class=VRCTorchDatasetV2,
        description="Blip2 Model V2"
    ),
    "mblip-mt0-v3": ModelVersionMap(
        processor_class=Blip2Processor,
        model_class=Blip2ForT5VQAV3,
        dataset_class=VRCTorchDatasetV2,
        description="Blip2 Model V3"
    )
}


MODEL_TYPES = list(MAP_CONFIG_TOKENIZER_MODEL_CLASS.keys())
ROOT_PATH = os.path.dirname(os.path.realpath(__file__))


def get_model_type_class(model_type: str) -> ModelVersionMap:
    """Lấy config class, tokenizer class, model class tương ứng với model_type."""
    if model_type is None:
        model_type = DEFAULT_MODEL_TYPE
    if model_type in MAP_CONFIG_TOKENIZER_MODEL_CLASS:
        return MAP_CONFIG_TOKENIZER_MODEL_CLASS[model_type]
    else:
        assert False, f"model_type must be in {MODEL_TYPES}"
