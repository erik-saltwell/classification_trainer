import torch
from datasets import Dataset

from classification_trainer.configuration import BaseModelInfo, DatasetInfo, TrainingInfo, TrainingLengthType
from classification_trainer.helpers.training_helper import create_trainer, load_base_model, run_training
from classification_trainer.protocols.logging_protocol import LoggingProtocol
from classification_trainer.utils import flush_gpu_memory
from classification_trainer.utils.common_paths import CommonPaths


def _next_batch_size_candidate(last_good: int, last_failed: int | None) -> int | None:
    if last_failed is None:
        # Phase 1: exponential doubling
        return 1 if last_good == 0 else last_good * 2
    else:
        # Phase 2: binary search between last_good and last_failed
        mid = (last_good + last_failed) // 2
        return None if mid == last_good else mid


def find_max_batch_size(
    dataset_info: DatasetInfo,
    base_model_info: BaseModelInfo,
    training_info: TrainingInfo,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    logger: LoggingProtocol,
) -> int:
    model, tokenizer = load_base_model(training_info)

    base_test_info: TrainingInfo = training_info.model_copy(
        update={
            "training_length_type": TrainingLengthType.STEPS,
            "gradient_accumulation_steps": 1,
            "per_device_batch_size": 1,
            "evaluation_enabled": True,
            "evaluation_steps": 1,
            "training_length": 3.0,
        }
    )

    last_good, last_failed = 0, None
    while (candidate := _next_batch_size_candidate(last_good, last_failed)) is not None:
        try:
            test_info = base_test_info.model_copy(update={"per_device_batch_size": candidate})
            trainer = create_trainer(
                dataset_info,
                test_info,
                model,
                tokenizer,
                train_dataset,
                False,
                eval_dataset=eval_dataset,
                output_dir=str(CommonPaths.get().get_model_checkpoint_directory(training_info.model_name)),
            )
            logger.report_message(f"Probing Batch Size: {candidate}")
            run_training(trainer, model)
            del trainer
            flush_gpu_memory()
            last_good = candidate
        except torch.cuda.OutOfMemoryError:
            flush_gpu_memory()
            last_failed = candidate

    del model, tokenizer
    flush_gpu_memory()

    if last_good == 0:
        logger.report_message("No batch size fits in GPU memory — even batch size 1 caused OOM.")
    else:
        logger.report_message(f"Largest successful batch size: {last_good}")

    return last_good
