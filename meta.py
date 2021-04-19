from transformers import AutoConfig

class Meta:
    config: AutoConfig

    def __init__(self, model_path):
        self.config = AutoConfig.from_pretrained(model_path)

    def get(self):
        return {
            'model': self.config.to_dict()
        }
