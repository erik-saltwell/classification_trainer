# Spec Decisions

## SFT Parameters in Training Info
The application will eventually be used to fine tune large language models.  As such, the training options need to take the following additional parameters:
- rank: int
- alpha_multiplier: int
- use_projection_modules: bool
- lora_dropout: float
- warmup_ratio: float
- learning_rate: float
- optim: str
- weight_decay: float
- lr_schedular_type: str

These should be specified in the training-info yaml file.  At the same time, we will eventually use these parameters to do training sweeps, and so they must be stored as a single SFTParameters object in the code so that wandb sweeps can treat these as our sweep options.

### lr_scheduler_type valid values
- "linear"
- "cosine"
- "cosine_with_restarts"
- "polynomial"
- "constant"
- "constant_with_warmup"
- "inverse_sqrt"
- "reduce_lr_on_plateau"
- "cosine_with_min_lr"
- "cosine_warmup_with_min_lr"
- "warmup_stable_decay"

### optim valid values
- "adamw_torch"
- "adamw_torch_fused"
- "adamw_torch_xla"
- "adamw_torch_npu_fused"
- "adamw_apex_fused"
- "adafactor"
- "adamw_anyprecision"
- "adamw_torch_4bit"
- "adamw_torch_8bit"
- "ademamix"
- "sgd"
- "adagrad"
- "adamw_bnb_8bit"
- "ademamix_8bit"
- "lion_8bit"
- "lion_32bit"
- "paged_adamw_32bit"
- "paged_adamw_8bit"
- "paged_ademamix_32bit"
- "paged_ademamix_8bit"
- "paged_lion_32bit"
- "paged_lion_8bit"
- "rmsprop"
- "rmsprop_bnb"
- "rmsprop_bnb_8bit"
- "rmsprop_bnb_32bit"
- "galore_adamw"
- "galore_adamw_8bit"
- "galore_adafactor"
- "galore_adamw_layerwise"
- "galore_adamw_8bit_layerwise"
- "galore_adafactor_layerwise"
- "lomo"
- "adalomo"
- "grokadamw"
- "schedule_free_radam"
- "schedule_free_adamw"
- "schedule_free_sgd"
- "apollo_adamw"
- "apollo_adamw_layerwise"
- "stable_adamw"


## Only instruct models are supported
Non-instruct (base) models require prompt templates to wrap inputs and outputs (training_fragment_id, eval_fragment_id), adding complexity with little benefit given the availability of strong instruct models. We support only instruct models, which handle prompting via their built-in chat template. BaseModelInfo therefore requires only `huggingface_name` and `chat_template`.

## Command Line Patterns
- Configuration data is stored in .yaml files, with different directories for dataset_info, base_model_info, training_info and similar.
- Command line commands take the names of yaml files (no ext) for each type of data needed, for example:
 - classification-trainer analyze-sequence-length dataset-info <dataset-yaml-file-without-extension>.
- yaml files are each stored in their own directory based on type of data.

## Supported Commands:
- analyze-sequence-length: requires dataset-info and base-model-info (for tokenzier)
- compute-batch-length: requires dataset-info, base-model-info and training-info
- training-run: requires dataset-info, base-model-info and training-info
- sweep: requires dataset-info, base-model-info and training-info
