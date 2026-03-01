from pydantic import BaseModel, ConfigDict


class InferenceInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Deterministic decoding for stable labels
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0

    # Keep completions short (pos/neg)
    max_new_tokens: int = 8

    # Optional: discourage rambling if your prompt isn't strict enough
    repetition_penalty: float | None = None

    # If you're using Unsloth, flip fast inference mode before generate
    prepare_unsloth_inference: bool = True
