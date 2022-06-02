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
    

weaviate_objects_batch = []
with open('/Users/raam/code/database_explorer/fraudTest.csv', newline='') as f:
    reader = csv.DictReader(f)
    for line in reader:
        fields = ["identifier","merchant", "category", "first", "last", "street", "city", "state", "job", ]
        trans = {}
        for f in fields:
            trans[f] = line[f]
        sentance_trans = copy.deepcopy(trans)
        del sentance_trans["identifier"]   
        trans["sentence"] = " ".join(sentance_trans.values())
        # print(trans["sentence"])    
        weaviate_objects_batch.append(trans)
        if len(weaviate_objects_batch) > 1000:
            print("break~")
            break
            
    

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