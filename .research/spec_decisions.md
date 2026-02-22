# Spec Decisions

## Only instruct models are supported
Non-instruct (base) models require prompt templates to wrap inputs and outputs (training_fragment_id, eval_fragment_id), adding complexity with little benefit given the availability of strong instruct models. We support only instruct models, which handle prompting via their built-in chat template. BaseModelInfo therefore requires only `huggingface_name` and `chat_template`.

# Command Line Patterns
- Configuration data is stored in .yaml files, with different directories for dataset_info, base_model_info, training_info and similar.
- Command line commands take the names of yaml files (no ext) for each type of data needed, for example:
 - classification-trainer analyze-sequence-length dataset-info <dataset-yaml-file-without-extension>.
- yaml files are each stored in their own directory based on type of data.

# Supported Commands:
- analyze-sequence-length: requires dataset-info and base-model-info (for tokenzier)
- compute-batch-length: requires dataset-info, base-model-info and training-info
- training-run: requires dataset-info, base-model-info and training-info
- sweep: requires dataset-info, base-model-info and training-info
