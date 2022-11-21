import os
import time
import subprocess
from multiprocessing import Process

import pytest
import requests
import uvicorn

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


def run_server():
    uvicorn.run(app)


@pytest.fixture(params=["t5-small",
                        "distilroberta-base",
                        "vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
                        "vblagoje/dpr-question_encoder-single-lfqa-wiki"], scope="function")
def server(request):
    os.environ["MODEL_NAME"] = request.param
    subprocess.call("python download.py", shell=True)
    proc = Process(target=run_server, args=(), daemon=True)
    proc.start()
    yield
    proc.kill()
    subprocess.call("rm -rf ./models", shell=True)


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

