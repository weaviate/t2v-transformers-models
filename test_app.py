## Run tests using test.sh file 
import os
import subprocess
import time
from multiprocessing import Process

import pytest
import requests
import uvicorn
import json
from app import app


def wait_for_uvicorn_start():
    url = 'http://localhost:8000/.well-known/ready'

    for i in range(0, 100):
        try:
            res = requests.get(url)
            if res.status_code == 204:
                return
            else:
                raise Exception(
                    "status code is {}".format(res.status_code))
        except Exception as e:
            print("Attempt {}: {}".format(i, e))
            time.sleep(2)

    raise Exception("did not start up")


@pytest.fixture(params=["t5-small",
                        "distilroberta-base",
                        "vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
                        "vblagoje/dpr-question_encoder-single-lfqa-wiki"], scope="function")
def server(request):
    os.environ["MODEL_NAME"] = request.param
    subprocess.call("python download.py", shell=True)
    f = open("./models/trans/config.json")
    config = json.load(f)
    # a simple check with high probability that this is the "actual" model.
    # If model doesn't exist download.py throws an error due to 404, if the model exists but is not 
    # optimized for setnencetransformer the model tries its best to autotune
    # Like
    # WARNING:root:No sentence-transformers model found with name /Users/raam/.cache/torch/sentence_transformers/distilroberta-base. Creating a new one with MEAN pooling.
    # Some weights of the model checkpoint at /Users/raam/.cache/torch/sentence_transformers/distilroberta-base were not used when initializing RobertaModel: ['lm_head.bias', 'lm_head.decoder.weight', 'lm_head.layer_norm.bias', 'lm_head.dense.weight', 'lm_head.dense.bias', 'lm_head.layer_norm.weight']
    #- This IS expected if you are initializing RobertaModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    #- This IS NOT expected if you are initializing RobertaModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    assert request.param.split("-")[-1] in config["_name_or_path"].split("-")[-1]
    proc = Process(target=run_server, args=(), daemon=True)
    proc.start()
    yield
    proc.kill()
    subprocess.call("rm -rf ./models", shell=True)


def run_server():
    uvicorn.run(app)

def test_vectorizing(server):
    wait_for_uvicorn_start()
    url = 'http://127.0.0.1:8000/vectors/'
    req_body = {'text': 'The London Eye is a ferris wheel at the River Thames.'}

    res = requests.post(url, json=req_body)
    resBody = res.json()
    vectorized_text = resBody['vector']

    assert 200 == res.status_code

    assert type(vectorized_text) is list

    # below tests that what we deem a reasonable vector is returned. We are
    # aware of 384 and 768 dim vectors, which should both fall in that
    # range
    assert 128 <= len(vectorized_text) <= 1024

    # now let's try two sentences

    req_body = {'text': 'The London Eye is a ferris wheel at the River Thames. Here is the second sentence.'}
    res = requests.post(url, json=req_body)
    resBody = res.json()
    vectorized_text = resBody['vector']

    assert 200 == res.status_code

    assert type(vectorized_text) is list

    assert 128 <= len(vectorized_text) <= 1024

