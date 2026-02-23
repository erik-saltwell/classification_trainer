from enum import Enum

import torch
from datasets import Dataset

from classification_trainer.configuration import BaseModelInfo, DatasetInfo, TrainingInfo, TrainingLengthType
from classification_trainer.helpers.training_helper import create_trainer, load_base_model, run_training
from classification_trainer.protocols.logging_protocol import LoggingProtocol
from classification_trainer.utils import flush_gpu_memory


class BatchSizeStrategy(Enum):
    INCREMENT_BY_1 = "increment_by_1"
    INCREMENT_BY_2 = "increment_by_2"
    INCREMENT_BY_5 = "increment_by_5"
    DOUBLE = "double"


def _next_batch_size(current: int, strategy: BatchSizeStrategy) -> int:
    match strategy:
        case BatchSizeStrategy.INCREMENT_BY_1:
            return current + 1
        case BatchSizeStrategy.INCREMENT_BY_2:
            return current + 2
        case BatchSizeStrategy.INCREMENT_BY_5:
            return current + 5
        case BatchSizeStrategy.DOUBLE:
            return current * 2


def find_max_batch_size(
    dataset_info: DatasetInfo,
    base_model_info: BaseModelInfo,
    training_info: TrainingInfo,
    train_dataset: Dataset,
    logger: LoggingProtocol,
    strategy: BatchSizeStrategy = BatchSizeStrategy.INCREMENT_BY_1,
    starting_batch_size: int = 1,
    max_batch_size: int | None = None,
) -> int:
    model, tokenizer = load_base_model(base_model_info, training_info)

    last_good = 0
    test_training_info: TrainingInfo = training_info.model_copy()
    test_training_info.training_length_type = TrainingLengthType.STEPS
    test_training_info.gradient_accumulation_steps = 1
    test_training_info.per_device_batch_size = starting_batch_size
    test_training_info.evaluation_enabled = True
    test_training_info.evaluation_steps = 1
    test_training_info.training_length = 2.0

    while True:
        try:
            trainer = create_trainer(
                dataset_info,
                test_training_info,
                base_model_info,
                model,
                tokenizer,
                train_dataset,
                False,
                eval_dataset=train_dataset,
            )
            logger.report_message(f"Probing Batch Size: {test_training_info.per_device_batch_size}")

            run_training(trainer)
            last_good = test_training_info.per_device_batch_size
            test_training_info.per_device_batch_size = _next_batch_size(
                test_training_info.per_device_batch_size, strategy
            )
            if max_batch_size is not None and test_training_info.per_device_batch_size > max_batch_size:
                logger.report_message(f"Reached max batch size cap ({max_batch_size}). Largest successful: {last_good}")
                del trainer, model, tokenizer
                flush_gpu_memory()
                return last_good
            del trainer
            flush_gpu_memory()
        except torch.cuda.OutOfMemoryError:
            logger.report_message(f"Largest succesfull batch size: {last_good}")
            del model, tokenizer
            flush_gpu_memory()
            return last_good
