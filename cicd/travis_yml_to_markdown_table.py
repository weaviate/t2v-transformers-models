#!/usr/bin/env python3

import yaml

print("|Model Name|Image Name|")
print("|---|---|")

with open(".travis.yml", 'r') as stream:
    try:
        travis = yaml.safe_load(stream)
        for model in travis['jobs']['include']:
            if model['stage'] != "buildanddeploy":
                continue

            model_name = model['env']['MODEL_NAME']
            tag = model_name
            if 'MODEL_TAG_NAME' in model['env']:
                tag = model['env']['MODEL_TAG_NAME']

            image_name = 'semitechnologies/transformers-inference:' + tag
            link = 'https://huggingface.co/' + model_name
            print("|`{}` ([Info]({}))|`{}`|".format(model_name, link, image_name))
    except yaml.YAMLError as exc:
        print(exc)
