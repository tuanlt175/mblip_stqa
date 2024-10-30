"""Configuration model mapping."""
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    # InstructBlipProcessor,
    # InstructBlipForConditionalGeneration
)
from icvrc.models.mblip_vqa import (
    Blip2ForVQA,
    Blip2ForVQAwST,
    Blip2ForVQAwQA,
    Blip2ForVQAwSTQA,
    Blip2ForVQAwQAQE,
    Blip2ForVQAwSTQAQE
)
from icvrc.mydatasets import VRCTorchDatasetForMBLIP, VRCTorchDatasetForMBLIPwST
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


DEFAULT_MODEL_TYPE = "mblip-vqa"
MAP_CONFIG_TOKENIZER_MODEL_CLASS = {
    "mblip-vqa": ModelVersionMap(
        processor_class=Blip2Processor,
        model_class=Blip2ForVQA,
        dataset_class=VRCTorchDatasetForMBLIP,
        description="mBlip Model"
    ),
    "mblip-vqa-st": ModelVersionMap(
        processor_class=Blip2Processor,
        model_class=Blip2ForVQAwST,
        dataset_class=VRCTorchDatasetForMBLIPwST,
        description="mBlip Model with Scene Text"
    ),
    "mblip-vqa-qa": ModelVersionMap(
        processor_class=Blip2Processor,
        model_class=Blip2ForVQAwQA,
        dataset_class=VRCTorchDatasetForMBLIP,
        description="mBlip Model with Question-Aware"
    ),
    "mblip-vqa-st-qa": ModelVersionMap(
        processor_class=Blip2Processor,
        model_class=Blip2ForVQAwSTQA,
        dataset_class=VRCTorchDatasetForMBLIPwST,
        description="mBlip Model with Scene Text and Question-Aware"
    ),
    "mblip-vqa-qa-qe": ModelVersionMap(
        processor_class=Blip2Processor,
        model_class=Blip2ForVQAwQAQE,
        dataset_class=VRCTorchDatasetForMBLIP,
        description="mBlip Model with Question-Aware and New Embedding for Qformer"
    ),
    "mblip-vqa-st-qa-qe": ModelVersionMap(
        processor_class=Blip2Processor,
        model_class=Blip2ForVQAwSTQAQE,
        dataset_class=VRCTorchDatasetForMBLIPwST,
        description="mBlip Model with Scene Text and Question-Aware (New Embedding for Qformer)"
    ),
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
