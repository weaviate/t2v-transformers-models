#!/usr/bin/env python3

import requests


print("|Model Name|Description|Image Name|")
print("|---|---|---|")

res = requests.get("https://configuration.semi.technology/v2/parameters/transformers_model?media_type=text&weaviate_version=v1.4.1&text_module=text2vec-transformers")
asJSON = res.json()

for opt in asJSON["options"]:
    name=opt["displayName"]
    description=opt["description"].replace('\n', '')
    image='semitechnologies/transformers-inference:' + opt["name"]
    if opt["name"] == "_custom":
        continue
    print(f"|{name}|{description}|{image}|")
