"""
InternVideo2_utils.py

InternVideo2 모델을 사용하는데 필요한 함수들을 적어두었습니다.
굳이, 안 읽어보셔도 됩니다.

바깥 코드에서는 load_video 함수만 사용합니다.

InternVideo2는 model, tokenizer만 반환하여 추론에 사용합니다.
- Llava-Video에서는 model, tokenizer, image_processor를 반환해서 이를 추론할 때 사용합니다.
- InternVideo2_5_Chat에서는 Generator만 반환하여 추론에 사용합니다.

함수 목록:
1. get_index
2. load_video
"""

import decord
import numpy as np
from decord import VideoReader, cpu
from torchvision import transforms

decord.bridge.set_bridge("torch")


def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array(
        [start + int(np.round(seg_size * idx)) for idx in range(num_segments)]
    )
    return offsets


def load_video(
    video_path,
    num_segments=8,
    return_msg=False,
    resolution=224,
    hd_num=4,
    padding=False,
):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.float().div(255.0)),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(mean, std),
        ]
    )

    frames = vr.get_batch(frame_indices)
    frames = frames.permute(0, 3, 1, 2)
    frames = transform(frames)

    T_, C, H, W = frames.shape

    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = (
            f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        )
        return frames, msg
    else:
        return frames
