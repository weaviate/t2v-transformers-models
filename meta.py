from transformers import AutoConfig


class Meta:
    config: AutoConfig

    def __init__(self, model_path: str, model_name: str, use_sentence_transformer_vectorizer: bool):
        if use_sentence_transformer_vectorizer:
            self.config = {"model_name": model_name, "model_type": None}
        else:
            self.config = AutoConfig.from_pretrained(model_path).to_dict()

    def get(self):
        return {
            'model': self.config
        }

    def get_model_type(self):
        return self.config['model_type']

    def get_architecture(self):
        architecture = None
        conf = self.config
        if "architectures" in conf:
            architecture = conf["architectures"][0]
        return architecture
