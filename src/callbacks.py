import os
from lightning.pytorch import Callback
from transformers import PreTrainedModel, AutoConfig
from huggingface_hub import HfApi, create_repo

from src.logger_config import logger

class PushToHuggingfaceCallback(Callback):
    def __init__(self, repo_owner=None, token=None, private=False, suffix: str="_rhetores"):
        super().__init__()

        self.token = token or os.getenv("HF_TOKEN", None)
        self.private = private
        self.api = HfApi()

        self.suffix = suffix

        self.repo_owner = repo_owner
        self.repo_id = None
        self.repo_name = None

        if self.token is None:
            raise ValueError("Hugging Face token is required. Set it as an environment variable HF_TOKEN or pass it to the callback.")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.repo_id is None or self.repo_name is None:
            self.repo_name = pl_module.model.pretrained_model_name_or_path.replace("/", "_")
            self.repo_name_id = f"{self.repo_owner}/{self.repo_name}" if self.repo_owner else self.repo_name
            self.repo_name_id = self.repo_name_id + self.suffix

            try:
                self.repo_id = self.api.create_repo(self.repo_name_id, exist_ok=True, token=self.token, private=True).repo_id
                # create_repo(self.repo_id, private=self.private, token=self.token)
            except Exception as e:
                print(f"Repository already exists or there was an error: {e}")

            # Ensure the model attribute of the Lightning module is a PreTrainedModel
            if not isinstance(pl_module.model.model, PreTrainedModel):
                raise ValueError("The 'model' attribute of your LightningModule must be an instance of PreTrainedModel")

        else:
            assert self.repo_id is not None
            assert self.repo_name is not None

        epoch = trainer.current_epoch

        path = trainer.logger.experiment.dir
        if path is None:
            path = trainer.default_root_dir

        path = os.path.join(path, "huggingface")

        #logger.info(f"PushToHuggingfaceCallback() -- Saving model to {path}")

        # saving the model and the corresponding tokenizer as huggingface models and tokenizer
        pl_module.model.tokenizer.save_pretrained(path)
        pl_module.model.model.save_pretrained(path)


        logger.info(f"PushToHuggingfaceCallback() -- Saved model and toeknizer to {path}")

        if pl_module.push_to_hub:
            self.api.upload_folder(
                commit_message=f"Pushing after epoch {epoch}",
                folder_path=path,
                repo_id=self.repo_id,
                repo_type="model",
                token=self.token,
                # ignore_patterns=[".gitattributes", ".gitignore", "**/checkpoints/**", "logs/**", "**/wandb/**"],
                # delete_patterns=[".gitattributes", ".gitignore", "**/checkpoints/**", "logs/**", "**/wandb/**"]
            )

            # push the model to the hub
            # pl_module.model.model.push_to_hub(
            #     repo_id=self.repo_id,
            #     commit_message=f"Upload model after epoch {epoch}",
            #     use_auth_token=self.token
            # )
        logger.info(f"Model uploaded to {self.repo_id} after epoch {epoch}")

    def on_exception(self, trainer, pl_module, exception):
        logger.info(f"An exception occurred: {exception}")
        # You might want to implement some error handling here