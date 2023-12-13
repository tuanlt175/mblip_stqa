import sys
import logging
import transformers
from pathlib import Path
import torch
from transformers import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    HfArgumentParser,
    set_seed,
    Trainer,
    DefaultDataCollator,
)
from transformers.training_args import TrainingArguments
from icvrc.arguments import (
    ModelArguments,
    DataTrainingArguments
)
from icvrc.model_map import get_model_type_class
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MAX_JOBS"] = "8"


logger = logging.getLogger(__name__)

PAD_LABEL_ID = -100


def main():
    """Main function."""
    # region Argument Parsing
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_class_map = get_model_type_class(model_args.model_type)

    if training_args.bf16:
        float_precision = torch.bfloat16
    elif training_args.fp16 or training_args.half_precision_backend:
        float_precision = torch.float16
    else:
        float_precision = torch.float32

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Sanity checks
    if training_args.output_dir is not None:
        training_args.output_dir = Path(training_args.output_dir)
        os.makedirs(training_args.output_dir, exist_ok=True)
    # endregion

    # region Checkpoints
    # Detecting last checkpoint.
    checkpoint = None
    if (
        len(os.listdir(training_args.output_dir)) > 0
        and not training_args.overwrite_output_dir
    ):
        config_path = training_args.output_dir / CONFIG_NAME
        weights_path = training_args.output_dir / WEIGHTS_NAME
        if config_path.is_file() and weights_path.is_file():
            checkpoint = training_args.output_dir
            logger.warning(
                f"Checkpoint detected, resuming training from checkpoint in {training_args.output_dir}. To avoid this"
                " behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        else:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to continue regardless."
            )

    # endregion

    # If passed along, set the training seed now.
    if training_args.seed is not None:
        set_seed(training_args.seed)

    # region Load pretrained model and tokenizer
    processor = model_class_map.processor_class.from_pretrained(model_args.model_name_or_path)
    # endregion

    # region Load datasets
    train_dataset = model_class_map.dataset_class(
        data_dir=data_args.data_folder,
        processor=processor,
        question_max_len=data_args.question_max_len,
        answer_max_len=data_args.answer_max_len,
        pad_label_id=-100,
        max_data_number=data_args.max_data_number,
        images_folder="training-images",
        data_file="preprocessed_vlsp2023_train.json",
    )
    hf_train_dataset = train_dataset.get_hf_dataset()

    eval_dataset = model_class_map.dataset_class(
        data_dir=data_args.data_folder,
        processor=processor,
        question_max_len=data_args.question_max_len,
        answer_max_len=data_args.answer_max_len,
        pad_label_id=-100,
        max_data_number=data_args.max_data_number,
        images_folder="dev-images",
        data_file="preprocessed_vlsp2023_dev.json",
    )
    hf_eval_dataset = eval_dataset.get_hf_dataset()

    # endregion

    # region Prepare model
    model_name_or_path = checkpoint or model_args.model_name_or_path
    model = model_class_map.model_class.from_pretrained(model_name_or_path, torch_dtype=float_precision)
    model.config.model_version = model_args.model_type
    if model_args.freeze_vision_model:
        for pname, param in model.vision_model.named_parameters():
            param.requires_grad = False
    if model_args.freeze_language_model:
        for pname, param in model.language_model.named_parameters():
            param.requires_grad = False
    # endregion
    data_collator = DefaultDataCollator()

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_eval_dataset,
        tokenizer=processor,
    )
    trainer.train()
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
