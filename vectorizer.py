from logging import Logger
import os
import math
from typing import Optional

import torch
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer, DPRContextEncoder, \
    DPRQuestionEncoder

# limit transformer batch size to limit parallel inference, otherwise we run
# into memory problems
MAX_BATCH_SIZE = 25  # TODO: take from config
DEFAULT_POOL_METHOD="masked_mean"

class VectorInputConfig(BaseModel):
    pooling_strategy: str


class VectorInput(BaseModel):
    text: str
    config: Optional[VectorInputConfig] = None

class Vectorizer:
    model: AutoModel
    tokenizer: AutoTokenizer
    cuda: bool
    cuda_core: str
    inferentia: bool
    model_type: str

    def __init__(self, model_path: str, cuda_support: bool, cuda_core: str, inferentia_support: bool,
                 model_type: str, architecture: str, logger: Logger):
        self.cuda = cuda_support
        self.cuda_core = cuda_core
        self.inferentia = inferentia_support
        self.model_type = model_type
        self.logger = logger
        # The max length is the max number of tokens, unfortunately in case of Neuron
        # cannot use dynamic size, which means that everything need to be padded up to
        # that max length. Which is why we set this differently for normal execution and
        # for Inferentia, as padding all to 500 would be wasteful 99% of the times.
        self.max_length = 128 if inferentia_support else 500

        self.model_delegate: HFModel = ModelFactory.model(model_type, architecture, inferentia_support,
                                                          logger, self.max_length)
        self.model = self.model_delegate.create_model(model_path)

        if self.cuda:
            self.model.to(self.cuda_core)
        self.model.eval() # make sure we're in inference mode, not training

        self.tokenizer = self.model_delegate.create_tokenizer(model_path)

    def tokenize(self, text:str):
        return self.tokenizer(text,
                              # for Inferentia support we need static-size inputs, so we pad to max_length
                              # see https://huggingface.co/blog/bert-inferentia-sagemaker#1-convert-your-hugging-face-transformer-to-aws-neuron
                              padding="max_length" if self.inferentia else True,
                              truncation=True, max_length=self.max_length,
                              add_special_tokens = True, return_tensors="pt")

    def get_embeddings(self, batch_results):
        return self.model_delegate.get_embeddings(batch_results)

    def get_batch_results(self, tokens, text):
        return self.model_delegate.get_batch_results(tokens, text)

    def pool_embedding(self, batch_results, tokens, config):
        return self.model_delegate.pool_embedding(batch_results, tokens, config)

    async def vectorize(self, text: str, config: VectorInputConfig):
        # set the number of Neuron cores to use when runnong in Inferentia
        if self.inferentia and ("NEURON_RT_NUM_CORES" not in os.environ or os.environ["NEURON_RT_NUM_CORES"] != "1"):
            # To use one neuron core per http worker - to scale a bit better (about 2-3%)
            os.environ["NEURON_RT_NUM_CORES"] = "1"

        with torch.no_grad():
            sentences = sent_tokenize(' '.join(text.split(),))
            num_sentences = len(sentences)
            number_of_batch_vectors = math.ceil(num_sentences / MAX_BATCH_SIZE)
            batch_sum_vectors = 0
            for i in range(0, number_of_batch_vectors):
                start_index = i * MAX_BATCH_SIZE
                end_index = start_index + MAX_BATCH_SIZE

                tokens = self.tokenize(sentences[start_index:end_index])
                if self.cuda:
                    tokens.to(self.cuda_core)
                batch_results = self.get_batch_results(tokens, sentences[start_index:end_index])
                batch_sum_vectors += self.pool_embedding(batch_results, tokens, config)
            return batch_sum_vectors.detach() / num_sentences


class HFModel:

    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None

    def create_tokenizer(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        return self.tokenizer

    def create_model(self, model_path):
        self.model = AutoModel.from_pretrained(model_path)
        return self.model

    def get_embeddings(self, batch_results):
        return batch_results[0]

    def get_batch_results(self, tokens, text):
        return self.model(**tokens)

    def pool_embedding(self, batch_results, tokens, config: VectorInputConfig):
        pooling_method = self.pool_method_from_config(config)
        if pooling_method == "cls":
            return self.get_embeddings(batch_results)[:, 0, :].sum(0)
        elif pooling_method == "masked_mean":
            return self.pool_sum(self.get_embeddings(batch_results), tokens['attention_mask'])
        else:
            raise Exception(f"invalid pooling method '{pooling_method}'")

    def pool_method_from_config(self, config: VectorInputConfig):
        if config is None:
            return DEFAULT_POOL_METHOD

        if config.pooling_strategy is None or config.pooling_strategy == "":
            return DEFAULT_POOL_METHOD

        return config.pooling_strategy

    def pool_sum(self, embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        sentences = sum_embeddings / sum_mask
        return sentences.sum(0)


class DPRModel(HFModel):

    def __init__(self, architecture: str):
        super().__init__()
        self.model = None
        self.architecture = architecture

    def create_model(self, model_path):
        if self.architecture == "DPRQuestionEncoder":
            self.model = DPRQuestionEncoder.from_pretrained(model_path)
        else:
            self.model = DPRContextEncoder.from_pretrained(model_path)
        return self.model

    def get_batch_results(self, tokens, text):
        return self.model(tokens['input_ids'], tokens['attention_mask'])

    def pool_embedding(self, batch_results, tokens, config: VectorInputConfig):
        # no pooling needed for DPR
        return batch_results["pooler_output"][0]


class T5Model(HFModel):

    def __init__(self, max_length: int):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.max_length = max_length

    def create_model(self, model_path):
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        return self.model

    def create_tokenizer(self, model_path):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        return self.tokenizer

    def get_embeddings(self, batch_results):
        return batch_results["encoder_last_hidden_state"]

    def get_batch_results(self, tokens, text):
        input_ids, attention_mask = tokens['input_ids'], tokens['attention_mask']

        target_encoding = self.tokenizer(
            text, padding="longest", max_length=self.max_length, truncation=True
        )
        labels = target_encoding.input_ids
        labels = torch.tensor(labels)

        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


class HFModelOnInferentia(HFModel):
    """A HuggingFace model running on the AWS Inferentia chip"""

    def __init__(self, logger: Logger, max_length: int):
        import torch_neuron

        super().__init__()
        self.model = None
        self.tokenizer = None
        self.logger = logger
        # where to store the compiled Neuron model before loading
        self.save_dir = "tmp-neuron-model"
        self.max_length = max_length

    def compile_model_for_neuron(self, model_path):
        """Compile the model for the Neuron architecture to allow to be used on the Inferentia chip
        See https://huggingface.co/blog/bert-inferentia-sagemaker#1-convert-your-hugging-face-transformer-to-aws-neuron
        """
        # get the original tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # create some dummy input padded up to the max length - to match the size used at inference
        dummy_input = "dummy input which will be padded later"
        embeddings = tokenizer(dummy_input, max_length=self.max_length, padding="max_length", return_tensors="pt")

        # load the original model first, so then it can be traced
        self.model = AutoModel.from_pretrained(model_path, torchscript=True)

        # compile the model wusing the dummy input for the number of neuron cores requested
        self.logger.info("Starting the tracing of the model for Neuron")
        model_neuron = torch.neuron.trace(self.model,
                                          example_inputs=(embeddings['input_ids'], embeddings['attention_mask']),
                                          compiler_args=['--neuroncore-pipeline-cores',
                                                         str(os.environ['NEURONCORE_PIPELINE_CORE'])],
                                          dynamic_batch_size=True)
        self.logger.info("Completed the tracing of the model for Neuron")

        # save the neuron model to be used later
        os.makedirs(self.save_dir, exist_ok=True)
        model_neuron.save(os.path.join(self.save_dir, "neuron_model.pt"))
        tokenizer.save_pretrained(self.save_dir)

        self.model.config.update({"traced_sequence_length": self.max_length})

    def create_tokenizer(self, model_path):
        # load the tokenizer we saved during the Neuron compile
        self.tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
        return self.tokenizer

    def create_model(self, model_path):
        """Compile the model for the Neuron architecture, then load it and return it"""
        # compile (trace) the model for Neuron
        self.compile_model_for_neuron(model_path)
        # load the traced model
        model = torch.jit.load(os.path.join(self.save_dir, "neuron_model.pt"))
        # use the original model's config
        model.config = self.model.config
        self.model = model
        return self.model

    def get_batch_results(self, tokens, text):
        return self.model(*(tokens['input_ids'], tokens['attention_mask']))


class ModelFactory:

    @staticmethod
    def model(model_type, architecture, inferentia_support, logger: Logger, max_length: int):
        if model_type == 't5':
            return T5Model(max_length)
        elif model_type == 'dpr':
            return DPRModel(architecture)
        elif inferentia_support:
            return HFModelOnInferentia(logger, max_length)
        else:
            return HFModel()
