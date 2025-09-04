IGNORE_INDEX = -100

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

SYSTEM_MESSAGE = "You are a helpful assistant."

GRPO_MESSAGE = "The user asks a question, and the assistant choose one option. The assistant first thinks about the reasoning process in the mind according! to! the! images! and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."

MULTIMODAL_KEYWORDS = ["pixel_values", 
                       "image_grid_thw", 
                       "video_grid_thw", 
                       "pixel_values_videos", 
                       "second_per_grid_ts", 
                       "depth_values",
                       "norm_values",
                       "flow_values",
                       "norm_value_grid"]