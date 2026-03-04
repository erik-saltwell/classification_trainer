from pydantic import BaseModel


class WandbConfig(BaseModel):
    project: str  # broad bucket: model/task
    sweep_name: str | None = None  # sweep container — stored as a wandb tag
    group: str | None = None  # campaign / batch of related runs
    job_type: str | None = None  # kind of run (e.g. "finetune", "eval")
