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
    # depth = [str(item) for item in droot.joinpath('nextqa_depth').joinpath(item['video']).iterdir()]
    # depth.sort()
    # flow = [str(item) for item in droot.joinpath('nextqa_flow').joinpath(item['video']).iterdir()]
    # flow.sort()
    # norm = [str(item) for item in droot.joinpath('nextqa_norm').joinpath(item['video']).iterdir()]
    # norm.sort()
    
    tmp_image= []
    for img in image:
        try:
            Image.open(img)
            tmp_image.append(img)
        except Exception as e:
            print(f"Find Error Image: {img}, {e}")
            continue
    image = tmp_image
    
    depth = []
    flow = []
    norm = []
    redundant_img = []
    for img in image:
        try:
            image_id = pathlib.Path(img).stem
            depth_path = droot.joinpath('nextqa_depth').joinpath(item['video']).joinpath(f'{image_id}_colored.png')
            flow_path = droot.joinpath('nextqa_flow').joinpath(item['video']).joinpath(f'{image_id}_pred_colored.png')
            norm_path = droot.joinpath('nextqa_norm').joinpath(item['video']).joinpath(f'{image_id}_pred_norm.png')
            Image.open(depth_path)
            Image.open(flow_path)
            Image.open(norm_path)
        except Exception as e:
            redundant_img.append(img)
            print(f"Find Error Image with id: {image_id}")
            continue
        depth.append(depth_path)
        flow.append(flow_path)
        norm.append(norm_path)

    if len(redundant_img) > 0:
        print("Redundant Images:", redundant_img)
        for rimg in redundant_img:
            image.remove(rimg)
    
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
    
with open("local_labels/train.json", 'w') as fout:
    json.dump(qwenvl_train, fout, indent=2)