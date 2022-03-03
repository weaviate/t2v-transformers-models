# transformers inference (for Weaviate)

This is the the inference container which is used by the Weaviate
`text2vec-transformers` module. You can download it directly from Dockerhub
using one of the pre-built images or built your own (as outlined below).

It is built in a way to support any PyTorch or Tensorflow transformers model,
either from the Huggingface Model Hub or from your disk.

This makes this an easy way to deploy your Weaviate-optimized transformers
NLP inference model to production using Docker or Kubernetes.

## Choose your model

### Pre-built images

You can download a selection of pre-built images directly from Dockerhub. We
have chosen publically available models that in our opinion are well suited for
semantic search. 

The pre-built models include:

|Model Name|Image Name|
|---|---|
|`distilbert-base-uncased` ([Info](https://huggingface.co/distilbert-base-uncased))|`semitechnologies/transformers-inference:distilbert-base-uncased`|
|`bert-base-uncased` ([Info](https://huggingface.co/bert-base-uncased))|`semitechnologies/transformers-inference:bert-base-uncased`|
|`distilroberta-base` ([Info](https://huggingface.co/distilroberta-base))|`semitechnologies/transformers-inference:distilroberta-base`|
|`sentence-transformers/stsb-distilbert-base` ([Info](https://huggingface.co/sentence-transformers/stsb-distilbert-base))|`semitechnologies/transformers-inference:sentence-transformers-stsb-distilbert-base`|
|`sentence-transformers/quora-distilbert-base` ([Info](https://huggingface.co/sentence-transformers/quora-distilbert-base))|`semitechnologies/transformers-inference:sentence-transformers-quora-distilbert-base`|
|`sentence-transformers/paraphrase-distilroberta-base-v1` ([Info](https://huggingface.co/sentence-transformers/paraphrase-distilroberta-base-v1))|`semitechnologies/transformers-inference:sentence-transformers-paraphrase-distilroberta-base-v1`|
|`kiri-ai/distiluse-base-multilingual-cased-et` ([Info](https://huggingface.co/kiri-ai/distiluse-base-multilingual-cased-et))|`semitechnologies/transformers-inference:kiri-ai-distiluse-base-multilingual-cased-et`|
|`sentence-transformers/msmarco-distilroberta-base-v2` ([Info](https://huggingface.co/sentence-transformers/msmarco-distilroberta-base-v2))|`semitechnologies/transformers-inference:sentence-transformers-msmarco-distilroberta-base-v2`|
|`sentence-transformers/msmarco-distilbert-base-v2` ([Info](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-v2))|`semitechnologies/transformers-inference:sentence-transformers-msmarco-distilbert-base-v2`|
|`sentence-transformers/stsb-xlm-r-multilingual` ([Info](https://huggingface.co/sentence-transformers/stsb-xlm-r-multilingual))|`semitechnologies/transformers-inference:sentence-transformers-stsb-xlm-r-multilingual`|
|`sentence-transformers/paraphrase-xlm-r-multilingual-v1` ([Info](https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1))|`semitechnologies/transformers-inference:sentence-transformers-paraphrase-xlm-r-multilingual-v1`|
|`sentence-transformers/gtr-t5-base` ([Info](https://huggingface.co/sentence-transformers/gtr-t5-base))|`semitechnologies/transformers-inference:sentence-transformers-gtr-t5-base`|
|`sentence-transformers/gtr-t5-large` ([Info](https://huggingface.co/sentence-transformers/gtr-t5-large))|`semitechnologies/transformers-inference:sentence-transformers-gtr-t5-large`|

The above image names always point to the latest version of the inference
container including the model. You can also make that explicit by appending
`-latest` to the image name. Additionally, you can pin the version to one of
the existing git tags of this repository. E.g. to pin `distilbert-base-uncased`
to version `1.0.0`, you can use
`semitechnologies/transformers-inference:distilbert-base-uncased-1.0.0`.

Your favorite model is not included? Open a pull-request to include it or build
a custom image as outlined below.

### Custom build with any huggingface model

You can build a docker image which supports any model from the huggingface
model hub with a two-line Dockerfile.

In the following example, we are going to build a custom image for the
`distilroberta-base` model.

Create a new `Dockerfile` (you do not need to clone this repository, any folder
on your machine is fine), we will name it `distilrobert.Dockerfile`. Add the
following lines to it:

```
FROM semitechnologies/transformers-inference:custom
RUN MODEL_NAME=distilroberta-base ./download.py
```

Now you just need to build and tag your Dockerfile, we will tag it as
`distilroberta-inference`:

```
docker build -f distilroberta.Dockerfile -t distilroberta-inference .
```

That's it! You can now push your image to your favorite registry or reference
it locally in your Weaviate `docker-compose.yaml` using the docker tag
`distilroberta-inference`.

### Custom build with a private / local model

You can build a docker image which supports any model which is compatible with
Huggingface's `AutoModel` and `AutoTokenzier`.

In the following example, we are going to build a custom image for a non-public
model which we have locally stored at `./my-model`.

Create a new `Dockerfile` (you do not need to clone this repository, any folder
on your machine is fine), we will name it `my-model.Dockerfile`. Add the
following lines to it:

```
FROM semitechnologies/transformers-inference:custom
COPY ./my-model /app/models/model
```

The above will make sure that your model end ups in the image at
`/app/models/model`. This path is important, so that the application can find the
model.

Now you just need to build and tag your Dockerfile, we will tag it as
`my-model-inference`:

```
docker build -f my-model.Dockerfile -t my-model-inference .
```

That's it! You can now push your image to your favorite registry or reference
it locally in your Weaviate `docker-compose.yaml` using the docker tag
`my-model-inference`.
