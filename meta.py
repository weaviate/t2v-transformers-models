import json
import os
from transformers import AutoConfig


class Meta:
    def __init__(
        self,
        model_path: str,
        model_name: str,
        use_sentence_transformer_vectorizer: bool,
        trust_remote_code: bool,
    ):
        self.model_path = model_path
        if use_sentence_transformer_vectorizer:
            if os.path.exists(f"{model_path}/model_config"):
                with open(f"{model_path}/model_config", "r") as f:
                    self.config = json.loads(f.read())
            else:
                self.config = {"model_name": model_name, "model_type": None}
        else:
            self.config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=trust_remote_code
            ).to_dict()

    def get(self):
        return {"model": self.config,
                "model_path": self.model_path,
        }
    
    

    def get_model_type(self):
        return self.config["model_type"]

    def get_architecture(self):
        architecture = None
        conf = self.config
        if "architectures" in conf:
            architecture = conf["architectures"][0]
        return architecture
