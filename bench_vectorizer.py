from vectorizer import Vectorizer, VectorInput
from meta import Meta
import asyncio
import time 
import csv 
import copy
meta_config = Meta('./models/model')
vec = Vectorizer('./models/model', False, "",
                     meta_config.getModelType(), meta_config.get_architecture(), True)



async def vectorize(text):
    return await vec.vectorize(text, None)
    
objects_batch = []
with open('[CSV]', newline='') as f:
    reader = csv.DictReader(f)
    for line in reader:
        pass
    ## Change me!
      
            
    
sentences = [obj["sentence"] for obj in weaviate_objects_batch]


tic = time.perf_counter()
print(tic)
print("Embedding started!")
results = []
for sen in sentences:
    res = asyncio.run(vectorize('who am i?'))
    results.append(res)
toc = time.perf_counter()
print(f"sentence embeddeings took {toc - tic:0.4f} seconds")