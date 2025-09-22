import copy
import os, sys
sys.path.append('/mnt/shared_workspace/pengjie/Qwen2-VL-Finetune')
sys.path.append('/mnt/shared_workspace/pengjie/Qwen2-VL-Finetune/src/dataset')
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset, DataLoader
import random
from src.params import DataArguments
from src.constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    SYSTEM_MESSAGE,
)
from tqdm import tqdm
from src.dataset.data_utils import get_image_info, get_video_info, llava_to_openai, pad_sequence

import pathlib
from PIL import Image

import numpy as np

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(SupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        is_video = False

        processor = self.processor
        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                images.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))

        elif "video" in sources:
            is_video = True
            images=None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"

            video_files = sources["video"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]

            videos = []
            for video_file in video_files:
                if not os.path.exists(video_file):
                    if not video_file.startswith("http"):
                        video_file = os.path.join(video_folder, video_file)
                video_input, video_kwargs = get_video_info(video_file, self.video_min_pixel, self.video_max_pixel, self.video_resized_w, self.video_resized_h, self.data_args.fps)
                videos.append(video_input)
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None

        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video))

        all_input_ids = [] 
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        # Qwen2-VL uses a default system message so I've added this.
        if len(SYSTEM_MESSAGE) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX) 
            
            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        for _, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
            gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"
            # print(user_input)
            if DEFAULT_IMAGE_TOKEN in user_input:
                inputs = processor(text=[user_input], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
            
            elif DEFAULT_VIDEO_TOKEN in user_input:
                if "Qwen2.5" in self.model_id:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt', **video_kwargs)
                    all_second_gird.extend(inputs["second_per_grid_ts"])
                else:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])

            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        # There is no need for eos or bos tokens in the input_ids
        # Qwen2-VL does not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        # eos_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        # input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id)

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird
        
        return data_dict

class SupervisedDatasetNextQA(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(SupervisedDatasetNextQA, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path
        # list_data_dict = list_data_dict[:100]
        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps
        self.image_folder = self.data_args.image_folder
        self.frame_length = self.data_args.frame_length
        # self.irregular_dir = pathlib.Path(self.image_folder).joinpath("irregular")
        # self.irregular_dir.mkdir(parents=True, exist_ok=True)
        # self.irregular_log = {}
        # for irregular_file in self.irregular_dir.iterdir():
        #     with open(irregular_file, "r") as f:
        #         self.irregular_log[irregular_file.stem] = json.load(f)

            
        # print("#"*40, type(self.processor))

    def check_and_sample(self, i):
        
        image_folder = pathlib.Path(self.data_args.image_folder)
        sources = self.list_data_dict[i]
        source_id = sources['id']
        image = sources['image']
        image_ids  = []
        for img in image:
            image_id = pathlib.Path(img).stem
            image_ids.append(image_id)
        
        image_root = pathlib.Path(sources['image'][0]).parent
        depth_root = pathlib.Path(sources['depth'][0]).parent
        flow_root = pathlib.Path(sources['flow'][0]).parent
        norm_root = pathlib.Path(sources['norm'][0]).parent
        
        selected_img_ids = random.sample(image_ids, self.frame_length)
        selected_image = [str(image_root.joinpath(f"{iid}.jpg")) for iid in selected_img_ids]
        selected_depth = [str(depth_root.joinpath(f"{iid}_colored.png")) for iid in selected_img_ids]
        selected_flow = [str(flow_root.joinpath(f"{iid}_pred_colored.png")) for iid in selected_img_ids]
        selected_norm = [str(norm_root.joinpath(f"{iid}_pred_norm.png")) for iid in selected_img_ids]
        return selected_image, selected_depth, selected_flow, selected_norm
        
        image.sort()
        
        image_root = pathlib.Path(sources['image'][0]).parent
        depth_root = pathlib.Path(sources['depth'][0]).parent
        flow_root = pathlib.Path(sources['flow'][0]).parent
        norm_root = pathlib.Path(sources['norm'][0]).parent
        
        if source_id in self.irregular_log:
            selected_img_ids = random.sample(image_ids, 4)
            selected_image = [str(image_root.joinpath(f"{iid}.jpg")) for iid in selected_img_ids]
            selected_depth = [str(depth_root.joinpath(f"{iid}_colored.png")) for iid in selected_img_ids]
            selected_flow = [str(flow_root.joinpath(f"{iid}_pred_colored.png")) for iid in selected_img_ids]
            selected_norm = [str(norm_root.joinpath(f"{iid}_pred_norm.png")) for iid in selected_img_ids]
            return selected_image, selected_depth, selected_flow, selected_norm
        irregular_log = {
            "image": [],
            "depth": [],
            "flow": [],
            "norm": []
        }
        irregular_id = []
        regular_id = []
        for img_id in image_ids:
            image_img = image_folder.joinpath(image_root.joinpath(f"{img_id}.jpg"))
            depth_img = image_folder.joinpath(depth_root.joinpath(f"{img_id}_colored.png"))
            flow_img = image_folder.joinpath(flow_root.joinpath(f"{img_id}_pred_colored.png"))
            norm_img = image_folder.joinpath(norm_root.joinpath(f"{img_id}_pred_norm.png"))
            # print(flow_img)
            if not image_img.exists() or not depth_img.exists() or not flow_img.exists() or not norm_img.exists():
                print(f"Find Error Image with id: {img_id}, {image_img.exists()}, {depth_img.exists()}, {flow_img.exists()}, {norm_img.exists()}")
                irregular_id.append(img_id)
                continue
            
            try:
                Image.open(image_img)
                Image.open(depth_img)
                Image.open(flow_img)
                Image.open(norm_img)
            except Exception as e:
                irregular_id.append(img_id)
                print(f"Find Error Image with id: {img_id}, {e}")
                continue
            regular_id.append(img_id)
            
        selected_img_ids = random.sample(regular_id, 4)
        selected_image = [str(image_root.joinpath(f"{iid}.jpg")) for iid in selected_img_ids]
        selected_depth = [str(depth_root.joinpath(f"{iid}_colored.png")) for iid in selected_img_ids]
        selected_flow = [str(flow_root.joinpath(f"{iid}_pred_colored.png")) for iid in selected_img_ids]
        selected_norm = [str(norm_root.joinpath(f"{iid}_pred_norm.png")) for iid in selected_img_ids]
        
            
        for iid in irregular_id:
            image_img = image_root.joinpath(f"{iid}.jpg")
            depth_img = depth_root.joinpath(f"{iid}_colored.png")
            float_img = flow_root.joinpath(f"{iid}_pred_colored.png")
            norm_img = norm_root.joinpath(f"{iid}_pred_norm.png")
            if image_img in self.list_data_dict[i]['image']:
                self.list_data_dict[i]['image'].remove(image_img)
            if depth_img in self.list_data_dict[i]['depth']:
                self.list_data_dict[i]['depth'].remove(depth_img)
            if float_img in self.list_data_dict[i]['flow']:
                self.list_data_dict[i]['flow'].remove(float_img)
            if norm_img in self.list_data_dict[i]['norm']:
                self.list_data_dict[i]['norm'].remove(norm_img)
            irregular_log["image"].append(str(image_img))
            irregular_log["depth"].append(str(depth_img))
            irregular_log["flow"].append(str(float_img))
            irregular_log["norm"].append(str(norm_img))
        with open(self.irregular_dir.joinpath(f"{source_id}.json"), "w") as f:
            json.dump(irregular_log, f, indent=4)
            
        return selected_image, selected_depth, selected_flow, selected_norm

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        # print(sources)
        is_video = False
        n_image = 0
        processor = self.processor
        selected_image, selected_depth, selected_flow, selected_norm = self.check_and_sample(i)
        # print(sources['norm'][0])
        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]
            images = []
            
            
            image_files = selected_image
            image_files.sort()
            
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                images.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))
                n_image = len(image_files)
        elif "video" in sources:
            is_video = True
            images=None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"

            video_files = sources["video"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]

            videos = []
            for video_file in video_files:
                if not os.path.exists(video_file):
                    if not video_file.startswith("http"):
                        video_file = os.path.join(video_folder, video_file)
                video_input, video_kwargs = get_video_info(video_file, self.video_min_pixel, self.video_max_pixel, self.video_resized_w, self.video_resized_h, self.data_args.fps)
                videos.append(video_input)
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None
        if "depth" in sources:
            image_files = sources["depth"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]
            depth = []
            image_files = selected_depth
            image_files.sort()
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                depth.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))

        if "flow" in sources:
            image_files = sources["flow"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]
            flow = []
            image_files = selected_flow
            image_files.sort()
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                flow.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))

        if "norm" in sources:
            image_files = sources["norm"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]
            norm = []
            image_files = selected_norm
            image_files.sort()
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                try:
                    norm.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))
                except:
                    print(image_file, "XXXXXX")
                    continue
        
        
        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video))

        all_input_ids = [] 
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []
        all_depth_values = []
        all_norm_values = []
        all_flow_values = []
        all_norm_grid = []
        n_norm_token = []

        # Qwen2-VL uses a default system message so I've added this.
        if len(SYSTEM_MESSAGE) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX) 
            
            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        for _, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
            gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"
            # print(gpt_response)
            if DEFAULT_IMAGE_TOKEN in user_input:
                inputs = processor(text=[user_input], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                
                tinput = processor(text=[user_input], images=depth, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                all_depth_values.append(tinput[pixel_key])
                tinput = processor(text=[user_input], images=norm, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                all_norm_values.append(tinput[pixel_key])
                all_norm_grid.append(tinput[grid_key])
                tinput = processor(text=[user_input], images=flow, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                all_flow_values.append(tinput[pixel_key])
            
            elif DEFAULT_VIDEO_TOKEN in user_input:
                if "Qwen2.5" in self.model_id:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt', **video_kwargs)
                    all_second_gird.extend(inputs["second_per_grid_ts"])
                else:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])

            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        # There is no need for eos or bos tokens in the input_ids
        # Qwen2-VL does not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        # eos_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        # input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id)

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            n_image=n_image
        )

        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw
            depth_values = torch.cat(all_depth_values, dim=0)
            data_dict["depth_values"] = depth_values
            norm_values = torch.cat(all_norm_values, dim=0)
            data_dict["norm_values"] = norm_values
            flow_values = torch.cat(all_flow_values, dim=0)
            data_dict["flow_values"] = flow_values
            data_dict["norm_value_grid"] = torch.cat(all_norm_grid, dim=0)

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird
        
        return data_dict



class SupervisedDatasetNextQAEval(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(SupervisedDatasetNextQAEval, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path
        # list_data_dict = list_data_dict[:10]
        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps
        self.frame_length = self.data_args.frame_length
        

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        # print(sources)
        qid = sources['id']
        is_video = False
        n_image = 0
        processor = self.processor
        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]
            images = []
            
            total_count = len(image_files)
            sample_count = self.frame_length
            indices = [int(i * (total_count - 1) / (sample_count - 1)) for i in range(sample_count)]
            indices = sorted(list(set(indices)))
            saved_files = []
            for i, file in enumerate(image_files):
                if i in indices:
                    saved_files.append(file)
            image_files = saved_files
            
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                images.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))
                n_image = len(image_files)
        elif "video" in sources:
            is_video = True
            images=None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"

            video_files = sources["video"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]

            videos = []
            for video_file in video_files:
                if not os.path.exists(video_file):
                    if not video_file.startswith("http"):
                        video_file = os.path.join(video_folder, video_file)
                video_input, video_kwargs = get_video_info(video_file, self.video_min_pixel, self.video_max_pixel, self.video_resized_w, self.video_resized_h, self.data_args.fps)
                videos.append(video_input)
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None
        if "depth" in sources:
            image_files = sources["depth"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]
            depth = []
            
            total_count = len(image_files)
            sample_count = self.frame_length
            indices = [int(i * (total_count - 1) / (sample_count - 1)) for i in range(sample_count)]
            indices = sorted(list(set(indices)))
            saved_files = []
            for i, file in enumerate(image_files):
                if i in indices:
                    saved_files.append(file)
            image_files = saved_files
            
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                depth.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))

        if "flow" in sources:
            image_files = sources["flow"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]
            flow = []
            
            total_count = len(image_files)
            sample_count = self.frame_length
            indices = [int(i * (total_count - 1) / (sample_count - 1)) for i in range(sample_count)]
            indices = sorted(list(set(indices)))
            saved_files = []
            for i, file in enumerate(image_files):
                if i in indices:
                    saved_files.append(file)
            image_files = saved_files
            
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                flow.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))

        if "norm" in sources:
            image_files = sources["norm"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]
            norm = []
            
            total_count = len(image_files)
            sample_count = self.frame_length
            indices = [int(i * (total_count - 1) / (sample_count - 1)) for i in range(sample_count)]
            indices = sorted(list(set(indices)))
            saved_files = []
            for i, file in enumerate(image_files):
                if i in indices:
                    saved_files.append(file)
            image_files = saved_files
            
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                norm.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))

            
        
        # print(sources['conversations'])
        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video))
        # print(sources)
        all_input_ids = [] 
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []
        all_depth_values = []
        all_norm_values = []
        all_flow_values = []
        all_norm_grid = []
        n_norm_token = []

        # Qwen2-VL uses a default system message so I've added this.
        if len(SYSTEM_MESSAGE) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX) 
            
            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        for _, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
            gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"
            
            if DEFAULT_IMAGE_TOKEN in user_input:
                inputs = processor(text=[user_input], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                
                tinput = processor(text=[user_input], images=depth, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                all_depth_values.append(tinput[pixel_key])
                tinput = processor(text=[user_input], images=norm, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                all_norm_values.append(tinput[pixel_key])
                all_norm_grid.append(tinput[grid_key])
                tinput = processor(text=[user_input], images=flow, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                all_flow_values.append(tinput[pixel_key])
            
            elif DEFAULT_VIDEO_TOKEN in user_input:
                if "Qwen2.5" in self.model_id:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt', **video_kwargs)
                    all_second_gird.extend(inputs["second_per_grid_ts"])
                else:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])

            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            input_ids = prompt_input_ids.squeeze(0)
            labels = response_input_ids.squeeze(0)

            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        # There is no need for eos or bos tokens in the input_ids
        # Qwen2-VL does not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        # eos_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        # input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id)

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            n_image=n_image,
            qid = qid
        )

        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw
            depth_values = torch.cat(all_depth_values, dim=0)
            data_dict["depth_values"] = depth_values
            norm_values = torch.cat(all_norm_values, dim=0)
            data_dict["norm_values"] = norm_values
            flow_values = torch.cat(all_flow_values, dim=0)
            data_dict["flow_values"] = flow_values
            data_dict["norm_value_grid"] = torch.cat(all_norm_grid, dim=0)

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird
        
        return data_dict



class SupervisedDatasetSQA3D(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(SupervisedDatasetSQA3D, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path
        # list_data_dict = list_data_dict[:100]
        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps
        self.image_folder = self.data_args.image_folder
        self.frame_length = self.data_args.frame_length


    def check_and_sample(self, i):
        
        image_folder = pathlib.Path(self.data_args.image_folder)
        sources = self.list_data_dict[i]
        source_id = sources['id']
        scene_id = source_id.split('_')[1:]
        scene_id = '_'.join(scene_id)
        image = sources['image']
        image_ids  = []
        for img in image:
            image_id = pathlib.Path(img).stem
            image_ids.append(image_id)
        
        image_root = image_folder.joinpath(pathlib.Path(sources['image'][0]).parent)
        depth_root = image_folder.joinpath(pathlib.Path(sources['depth'][0]).parent)
        norm_root = image_folder.joinpath(pathlib.Path(sources['norm'][0]).parent)
        pc_root = image_folder.joinpath(pathlib.Path(sources['pc'][0]))
        pc_feature_root = image_folder.joinpath(pathlib.Path(sources['pc_feature'][0]))
        if len(image_ids) <= self.frame_length:
            selected_img_ids = image_ids
        else:
            selected_img_ids = random.sample(image_ids, self.frame_length)
        selected_image = [str(image_root.joinpath(f"{iid}.jpg")) for iid in selected_img_ids]
        selected_depth = [str(depth_root.joinpath(f"{iid}.png")) for iid in selected_img_ids]
        selected_norm = [str(norm_root.joinpath(f"{iid}_pred_norm.png")) for iid in selected_img_ids]
        selected_pc = str(pc_root)
        selected_pc_feature = str(pc_feature_root)
        return selected_image, selected_depth, selected_norm, selected_pc, selected_pc_feature
        

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        # print(sources)
        is_video = False
        n_image = 0
        processor = self.processor
        selected_image, selected_depth, selected_norm, selected_pc, selected_pc_feature = self.check_and_sample(i)
        # print(sources['norm'][0])
        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]
            images = []
            
            
            image_files = selected_image
            image_files.sort()
            
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                images.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))
                n_image = len(image_files)
        elif "video" in sources:
            is_video = True
            images=None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"

            video_files = sources["video"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]

            videos = []
            for video_file in video_files:
                if not os.path.exists(video_file):
                    if not video_file.startswith("http"):
                        video_file = os.path.join(video_folder, video_file)
                video_input, video_kwargs = get_video_info(video_file, self.video_min_pixel, self.video_max_pixel, self.video_resized_w, self.video_resized_h, self.data_args.fps)
                videos.append(video_input)
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None
        if "depth" in sources:
            image_files = sources["depth"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]
            depth = []
            image_files = selected_depth
            image_files.sort()
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                depth.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))

        if "norm" in sources:
            image_files = sources["norm"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]
            norm = []
            image_files = selected_norm
            image_files.sort()
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                try:
                    norm.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))
                except:
                    print(image_file, "XXXXXX")
                    continue
        
        if 'pc' in sources and 'pc_feature' in sources:
            pc_path = selected_pc
            pc_feature_path = selected_pc_feature
            pc_feat = torch.load(pc_feature_path, map_location="cpu")  # [N, 1408]
            if isinstance(pc_feat, np.ndarray):
                pc_feat = torch.tensor(pc_feat).float()
            pc = np.load(pc_path)
            pc = torch.tensor(pc).float().cpu()
            # sample 10000 points: [N, 1408] -> [10000, 1408]
            if pc_feat.shape[0] > 5000:
                idxes = torch.sort(torch.randperm(pc_feat.shape[0])[:5000])[1]
                pc_feat = pc_feat[idxes]
                pc = pc[idxes]
            else:
                pc_feat = torch.cat([pc_feat, torch.zeros(5000 - pc_feat.shape[0], 1408)], dim=0)
                pc = torch.cat([pc, torch.zeros(5000 - pc.shape[0], 3)], dim=0)
        
        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video))

        all_input_ids = [] 
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []
        all_depth_values = []
        all_norm_values = []
        all_flow_values = []
        all_norm_grid = []
        n_norm_token = []

        # Qwen2-VL uses a default system message so I've added this.
        if len(SYSTEM_MESSAGE) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX) 
            
            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        for _, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
            gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"
            # print(gpt_response)
            if DEFAULT_IMAGE_TOKEN in user_input:
                inputs = processor(text=[user_input], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                
                tinput = processor(text=[user_input], images=depth, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                all_depth_values.append(tinput[pixel_key])
                tinput = processor(text=[user_input], images=norm, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                all_norm_values.append(tinput[pixel_key])
                all_norm_grid.append(tinput[grid_key])
                # tinput = processor(text=[user_input], images=flow, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                # all_flow_values.append(tinput[pixel_key])
            
            elif DEFAULT_VIDEO_TOKEN in user_input:
                if "Qwen2.5" in self.model_id:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt', **video_kwargs)
                    all_second_gird.extend(inputs["second_per_grid_ts"])
                else:
                    inputs = processor(text=[user_input], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])

            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        # There is no need for eos or bos tokens in the input_ids
        # Qwen2-VL does not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        # eos_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        # input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id)

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            n_image=n_image
        )
        data_dict['pc_values'] = pc
        data_dict['pc_feature_values'] = pc_feat
        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw
            depth_values = torch.cat(all_depth_values, dim=0)
            data_dict["depth_values"] = depth_values
            norm_values = torch.cat(all_norm_values, dim=0)
            data_dict["norm_values"] = norm_values
            # flow_values = torch.cat(all_flow_values, dim=0)
            # data_dict["flow_values"] = flow_values
            data_dict["norm_value_grid"] = torch.cat(all_norm_grid, dim=0)

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird
        
        return data_dict
    
    
class SFTDatasetSQA3DEval(Dataset):
    """Dataset for DPO training"""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(SFTDatasetSQA3DEval, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps
        self.image_folder = self.data_args.image_folder
        self.frame_length = self.data_args.frame_length

    def __len__(self):
        return len(self.list_data_dict)
    
    def check_and_sample(self, i):
        
        image_folder = pathlib.Path(self.data_args.image_folder)
        sources = self.list_data_dict[i]
        source_id = sources['id']
        scene_id = source_id.split('_')[1:]
        scene_id = '_'.join(scene_id)
        image = sources['image']
        image_ids  = []
        for img in image:
            image_id = pathlib.Path(img).stem
            image_ids.append(image_id)
        
        image_root = image_folder.joinpath(pathlib.Path(sources['image'][0]).parent)
        depth_root = image_folder.joinpath(pathlib.Path(sources['depth'][0]).parent)
        norm_root = image_folder.joinpath(pathlib.Path(sources['norm'][0]).parent)
        pc_root = image_folder.joinpath(pathlib.Path(sources['pc'][0]))
        pc_feature_root = image_folder.joinpath(pathlib.Path(sources['pc_feature'][0]))
        
        selected_img_ids = random.sample(image_ids, self.frame_length)
        selected_image = [str(image_root.joinpath(f"{iid}.jpg")) for iid in selected_img_ids]
        selected_depth = [str(depth_root.joinpath(f"{iid}.png")) for iid in selected_img_ids]
        selected_norm = [str(norm_root.joinpath(f"{iid}_pred_norm.png")) for iid in selected_img_ids]
        selected_pc = str(pc_root)
        selected_pc_feature = str(pc_feature_root)
        return selected_image, selected_depth, selected_norm, selected_pc, selected_pc_feature
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = copy.deepcopy(self.list_data_dict[i])
        pid = sources['id']
        is_video = False

        contents = []
        selected_image, selected_depth, selected_norm, selected_pc, selected_pc_feature = self.check_and_sample(i)

        if "image" in sources:

            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]
            
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                contents.append(get_image_content(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))

        elif "video" in sources:
            is_video = True

            video_files = sources["video"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]

            for video_file in video_files:
                if not os.path.exists(video_file):
                    if not video_file.startswith("http"):
                        video_file = os.path.join(video_folder, video_file)
                contents.append(get_video_content(video_file, self.video_min_pixel, self.video_max_pixel, self.video_resized_w, self.video_resized_h, self.data_args.fps))

        if "depth" in sources:
            image_files = sources["depth"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]
            depth = []

            
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                contents.append(get_multimodal_content(image_file, 
                                                    self.image_min_pixel, 
                                                    self.image_max_pixel, 
                                                    self.image_resized_w, 
                                                    self.image_resized_h,
                                                    'depth'))
        if "norm" in sources:
            image_files = sources["norm"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]
            norm = []
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                try:
                    contents.append(get_multimodal_content(image_file, 
                                                    self.image_min_pixel, 
                                                    self.image_max_pixel, 
                                                    self.image_resized_w, 
                                                    self.image_resized_h,
                                                    'norm'))
                except:
                    print(image_file, "XXXXXX")
                    continue
        if 'pc' in sources and 'pc_feature' in sources:
            pc_path = selected_pc
            pc_feature_path = selected_pc_feature
            pc_feat = torch.load(pc_feature_path, map_location="cpu")  # [N, 1408]
            if isinstance(pc_feat, np.ndarray):
                pc_feat = torch.tensor(pc_feat).float()
            pc = np.load(pc_path)
            pc = torch.tensor(pc).float().cpu()
            # sample 10000 points: [N, 1408] -> [10000, 1408]
            if pc_feat.shape[0] > 5000:
                idxes = torch.sort(torch.randperm(pc_feat.shape[0])[:5000])[1]
                pc_feat = pc_feat[idxes]
                pc = pc[idxes]
            else:
                pc_feat = torch.cat([pc_feat, torch.zeros(5000 - pc_feat.shape[0], 1408)], dim=0)
                pc = torch.cat([pc, torch.zeros(5000 - pc.shape[0], 3)], dim=0)
            contents.append(
                {'type': 'pc',
                'pc': pc,
                'pc_feature': pc_feat}
            )
        # print(sources['conversations'])
        # sources['conversations'][0]['value'] += f"\n{GRPO_MESSAGE}\n"
        sources['conversations'][1]['value'] = sources['conversations'][1]['value'].strip() + f"{DEFAULT_IM_END_TOKEN}\n"
        # print(sources['conversations'][0]['value'])
        conversations = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=False))
        # print(i, conversations)
        user_input = conversations[0]
        gpt_response = conversations[1]

        text_content = {"type": "text", "text": user_input['content']}

        contents.append(text_content)

        user_prompt = [{"role": "user", "content": contents}]

        if len(SYSTEM_MESSAGE) > 0:
            system_message = {"role": "system", "content": SYSTEM_MESSAGE}
            user_prompt.insert(0, system_message)
        
        data_dict = dict(
            prompt=user_prompt,
            assistant=gpt_response,
            qid=pid
        )

        return data_dict

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []
        batch_n_image = []
        batch_depth_value = []
        batch_norm_value = []
        batch_flow_value = []
        batch_norm_value_grid = []
        batch_qid = []
        batch_pc = []
        batch_pc_feature = []
        
        
        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])
                batch_depth_value.append(example["depth_values"])
                batch_norm_value.append(example["norm_values"])
                batch_norm_value_grid.append(example["norm_value_grid"])
                
            if "flow_values" in example:
                batch_flow_value.append(example["flow_values"])
                
            if 'qid' in example:
                batch_qid.append(example['qid'])
            if 'pc_values' in example:
                batch_pc.append(example['pc_values'])
            if 'pc_feature_values' in example:
                batch_pc_feature.append(example['pc_feature_values'])
                
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])
            
            if "n_image" in keys:
                batch_n_image.append(example['n_image'])
        
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        if len(batch_n_image) > 0:
            data_dict["n_image"] = batch_n_image
        if len(batch_depth_value) > 0:
            depth_values = torch.cat(batch_depth_value, dim=0)
            data_dict["depth_values"] = depth_values
        if len(batch_norm_value) > 0:
            norm_values = torch.cat(batch_norm_value, dim=0)
            data_dict["norm_values"] = norm_values
        if len(batch_flow_value) > 0:
            flow_values = torch.cat(batch_flow_value, dim=0)
            data_dict["flow_values"] = flow_values
        if len(batch_norm_value_grid) > 0:
            norm_value_grid = torch.cat(batch_norm_value_grid, dim=0)
            data_dict["norm_value_grid"] = norm_value_grid
        
        if len(batch_qid) > 0:
            data_dict["qid"] = batch_qid
            
        if len(batch_pc) > 0:
            pc_values = torch.stack(batch_pc)
            data_dict["pc_values"] = pc_values
            
        if len(batch_pc_feature) > 0:
            batch_pc_feature = torch.stack(batch_pc_feature)
            data_dict["pc_feature_values"] = batch_pc_feature
            
        return data_dict
    
    
def get_image_content(image_path, min_pixel, max_pixel, width, height):
    # Using this because of process_vision_info function
    # Need to fix this in the future
    content = {
        "type": "image", 
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height

    return content
    
def get_multimodal_content(image_path, min_pixel, max_pixel, width, height, multimodal_type='image'):
    # Using this because of process_vision_info function
    # Need to fix this in the future
    content = {
        "type": multimodal_type, 
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height

    return content

    
class SFTDatasetNextQAEval2(Dataset):
    """Dataset for DPO training"""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(SFTDatasetNextQAEval2, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps

    def __len__(self):
        return len(self.list_data_dict)
    
    def check_and_sample(self, i):
        
        image_folder = pathlib.Path(self.data_args.image_folder)
        sources = self.list_data_dict[i]
        source_id = sources['id']
        image = sources['image']
        image_ids  = []
        for img in image:
            image_id = pathlib.Path(img).stem
            image_ids.append(image_id)
        
        image_root = pathlib.Path(sources['image'][0]).parent
        depth_root = pathlib.Path(sources['depth'][0]).parent
        flow_root = pathlib.Path(sources['flow'][0]).parent
        norm_root = pathlib.Path(sources['norm'][0]).parent
        
        selected_img_ids = random.sample(image_ids, 10)
        selected_image = [str(image_root.joinpath(f"{iid}.jpg")) for iid in selected_img_ids]
        selected_depth = [str(depth_root.joinpath(f"{iid}_colored.png")) for iid in selected_img_ids]
        selected_flow = [str(flow_root.joinpath(f"{iid}_pred_colored.png")) for iid in selected_img_ids]
        selected_norm = [str(norm_root.joinpath(f"{iid}_pred_norm.png")) for iid in selected_img_ids]
        return selected_image, selected_depth, selected_flow, selected_norm
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = copy.deepcopy(self.list_data_dict[i])
        pid = sources['id']
        is_video = False

        contents = []
        selected_image, selected_depth, selected_flow, selected_norm = self.check_and_sample(i)

        if "image" in sources:

            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]
                
            # total_count = len(image_files)
            # sample_count = 10
            # indices = [int(i * (total_count - 1) / (sample_count - 1)) for i in range(sample_count)]
            # indices = sorted(list(set(indices)))
            
            # indices = [i for i in range(total_count)]
            
            # saved_files = []
            # for i, file in enumerate(image_files):
            #     if i in indices:
            #         saved_files.append(file)
            # image_files = saved_files
            
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                contents.append(get_image_content(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))

        elif "video" in sources:
            is_video = True

            video_files = sources["video"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]

            for video_file in video_files:
                if not os.path.exists(video_file):
                    if not video_file.startswith("http"):
                        video_file = os.path.join(video_folder, video_file)
                contents.append(get_video_content(video_file, self.video_min_pixel, self.video_max_pixel, self.video_resized_w, self.video_resized_h, self.data_args.fps))

        if "depth" in sources:
            image_files = sources["depth"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]
            depth = []
            # total_count = len(image_files)
            # sample_count = 10
            # indices = [int(i * (total_count - 1) / (sample_count - 1)) for i in range(sample_count)]
            # indices = sorted(list(set(indices)))
            
            # indices = [i for i in range(total_count)]
            
            # saved_files = []
            # for i, file in enumerate(image_files):
            #     if i in indices:
            #         saved_files.append(file)
            # image_files = saved_files
            
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                contents.append(get_multimodal_content(image_file, 
                                                    self.image_min_pixel, 
                                                    self.image_max_pixel, 
                                                    self.image_resized_w, 
                                                    self.image_resized_h,
                                                    'depth'))
        if "flow" in sources:
            image_files = sources["flow"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]
            flow = []
            # total_count = len(image_files)
            # sample_count = 10
            # indices = [int(i * (total_count - 1) / (sample_count - 1)) for i in range(sample_count)]
            # indices = sorted(list(set(indices)))
            
            # indices = [i for i in range(total_count)]
            
            # saved_files = []
            # for i, file in enumerate(image_files):
            #     if i in indices:
            #         saved_files.append(file)
            # image_files = saved_files
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                contents.append(get_multimodal_content(image_file, 
                                                    self.image_min_pixel, 
                                                    self.image_max_pixel, 
                                                    self.image_resized_w, 
                                                    self.image_resized_h,
                                                    'flow'))
        if "norm" in sources:
            image_files = sources["norm"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]
            norm = []
            # total_count = len(image_files)
            # sample_count = 10
            # indices = [int(i * (total_count - 1) / (sample_count - 1)) for i in range(sample_count)]
            # indices = sorted(list(set(indices)))
            
            # indices = [i for i in range(total_count)]
            
            # saved_files = []
            # for i, file in enumerate(image_files):
            #     if i in indices:
            #         saved_files.append(file)
            # image_files = saved_files
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                try:
                    contents.append(get_multimodal_content(image_file, 
                                                    self.image_min_pixel, 
                                                    self.image_max_pixel, 
                                                    self.image_resized_w, 
                                                    self.image_resized_h,
                                                    'norm'))
                except:
                    print(image_file, "XXXXXX")
                    continue
        # print(sources['conversations'])
        # sources['conversations'][0]['value'] += f"\n{GRPO_MESSAGE}\n"
        sources['conversations'][1]['value'] = sources['conversations'][1]['value'].strip() + f"{DEFAULT_IM_END_TOKEN}\n"
        # print(sources['conversations'][0]['value'])
        conversations = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=False))
        # print(i, conversations)
        user_input = conversations[0]
        gpt_response = conversations[1]

        text_content = {"type": "text", "text": user_input['content']}

        contents.append(text_content)

        user_prompt = [{"role": "user", "content": contents}]

        if len(SYSTEM_MESSAGE) > 0:
            system_message = {"role": "system", "content": SYSTEM_MESSAGE}
            user_prompt.insert(0, system_message)
        
        data_dict = dict(
            prompt=user_prompt,
            assistant=gpt_response,
            qid=pid
        )

        return data_dict
    
    
class DataCollatorForSFTEvalDataset2(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self):
        pass

    def __call__(self, examples):
        return examples
        
    
def make_supervised_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDatasetNextQA(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    )
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)
    
def make_supervised_sqa3d_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDatasetSQA3D(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    )
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)
    
def make_supervised_eval_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SFTDatasetNextQAEval2(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    )
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)
    
def make_supervised_eval_data_module2(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SFTDatasetNextQAEval2(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    )
    data_collator = DataCollatorForSFTEvalDataset2()

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)
    
    
def make_supervised_eval_sqa3d_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SFTDatasetSQA3DEval(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    )
    data_collator = DataCollatorForSFTEvalDataset2()

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)

if __name__ == "__main__":
    from src.model.qwenvl_more_modality import Qwen2_5_VLForConditionalGenerationMore, assign_qformer, Qwen2_5_VLProcessorOneToken, assign_prefusion
    data_config = DataArguments(
        data_path = 'local_labels/train_filtered.json',
        image_folder = '.',
        image_min_pixels = 256*28*28,
        image_max_pixels=256*28*28,
        image_resized_width=112,
        image_resized_height=112,
    )
    
    processor = Qwen2_5_VLProcessorOneToken.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct', 
                                                            n_frames = 16*4)
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)
    # dataset = SupervisedDatasetNextQA(
    #     'local_labels/train_filtered.json',
    #     processor,
    #     data_config,
    #     'Qwen/Qwen2.5-VL-3B-Instruct'
    # )
    
    dataset = SupervisedDatasetNextQAEval(
        'local_labels/val.json',
        processor,
        data_config,
        'Qwen/Qwen2.5-VL-3B-Instruct'
    )
    
    dataloader = DataLoader(dataset, batch_size=16, num_workers=32, collate_fn=data_collator)
    
    for batch in tqdm(dataloader):
        print(batch.keys())