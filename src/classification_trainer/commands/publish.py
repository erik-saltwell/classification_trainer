from __future__ import annotations

from attr import dataclass

from classification_trainer.configuration import PublishingInfo, TrainingInfo
from classification_trainer.helpers import publishing_helper
from classification_trainer.protocols import CommmandProtocol, LoggingProtocol


@dataclass
class PublishCommand(CommmandProtocol):
    training_info: TrainingInfo
    publishing_info: PublishingInfo

    def execute(self, logger: LoggingProtocol) -> None:
        logger.report_message(
            f"[blue]Publishing {self.training_info.model_name} artifacts to HuggingFace Hub...[/blue]"
        )
        publishing_helper.publish_model(self.training_info, self.publishing_info, logger)
        logger.report_message("[green]All formats published successfully.[/green]")
