import pathlib
import json 
import shutil
from tqdm import tqdm
import os
from PIL import Image

root_path = pathlib.Path("nextqa_annotation")
training_subset = root_path.joinpath("train.json")
val_subset = root_path.joinpath("val.json")


with open('nextqa/nextqa_annotation/train.json', 'r') as fin:
    training_json = json.load(fin)
with open('nextqa/nextqa_annotation/val.json', 'r') as fin:
    val_json = json.load(fin)
# print(training_json[0])
from tqdm import tqdm

qwenvl_train = []
droot = pathlib.Path("nextqa")
for item in tqdm(training_json):
    qid = item['qid']
    image = [str(item) for item in droot.joinpath('nextqa_frame').joinpath(item['video']).iterdir()]
    image.sort()
    depth = [str(item) for item in droot.joinpath('nextqa_depth').joinpath(item['video']).iterdir()]
    depth.sort()
    flow = [str(item) for item in droot.joinpath('nextqa_flow').joinpath(item['video']).iterdir()]
    flow.sort()
    norm = [str(item) for item in droot.joinpath('nextqa_norm').joinpath(item['video']).iterdir()]
    norm.sort()
    
    
    answer_options = [item[f"a{i}"] for i in range(item['num_option'])]
    answer_options = [f"a{i}: {answer_options[i]}" for i in range(item['num_option'])]
    answer_options_str = '\n'.join(answer_options)
    # print(answer_options)
    conversations = [
        {
            "from": 'human',
            'value': f"<image>\n{item['question']} Choose one of the following options: \n{answer_options_str}"
        },
        {
            "from": 'gpt',
            'value': f"{item['answer']}, {answer_options[item['answer']]}"
        }
    ]
    # print(image, depth, flow, norm)
    qwenvl_train.append(
        {
            "id": qid,
            "image": image,
            "depth": depth,
            "flow": flow,
            "norm": norm,
            "conversations": conversations
        }
    )
    if len(qwenvl_train) % 5000 == 0:
        print(f"Processed {len(qwenvl_train)} samples")
    
with open("local_labels/train.json", 'w') as fout:
    json.dump(qwenvl_train, fout, indent=2)