import os
from logging import getLogger
from fastapi import FastAPI, Response, status
from transformer import Transformer, TextInput, TextsInput


app = FastAPI()
logger = getLogger('uvicorn')

@app.on_event("startup")
def startup_event():
    global encoder
    model_path = "/./models/trans" #we assume the model was already downloaded
    device = os.environ.get("DEVICE") # device can be any pytorch device. 
    # Current implementation should detect cuda devices automatically
    encoder = Transformer().load_model(model_path, device)


    
@app.get("/.well-known/live", response_class=Response)
@app.get("/.well-known/ready", response_class=Response)
def live_and_ready(response: Response):
    response.status_code = status.HTTP_204_NO_CONTENT


@app.post("/texts/") #future support for /medias/
# Example Input JSON
# '{"text": ["cats are better than dogs", "and better than humans"]}'
def read_items(item: TextsInput, response: Response):
    return read_item(item, response)

@app.post("/vectors/")
# Example Input JSON
# '{"text": "cats are better than dogs"}'
def read_item(item: TextInput, response: Response):
    try:
        vector = encoder(item.text)
        return {"text": item.text, "vector": vector.tolist(), "dim": len(vector)}
    except Exception as e:
        logger.exception('Something went wrong while vectorizing data.')
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": str(e)}
