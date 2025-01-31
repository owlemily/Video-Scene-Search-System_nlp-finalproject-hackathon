"""
InternVideo2_5_Chat_utils.py

InternVideo2_5_Chat_8B 모델을 사용하는데 필요한 함수들을 적어두었습니다.
바깥 코드에서는 DescriptionGenerator 클래스를 사용하여 바로 추론합니다.
클래스를 상세히 보실 필요는 없고, 바깥 코드에서는 DescriptionGenerator만 사용하시면 됩니다. (필요시, 현재 파이썬 파일의 main 함수 참고)

InternVideo2_5_Chat에서는 Generator만 반환하여 추론에 사용합니다.
- InternVideo2는 model, tokenizer만 반환했었음
- Llava-Video에서는 model, tokenizer, image_processor를 반환해서 이를 추론할 때 사용합니다.

클래스 목록:
1. DescriptionGenerator
"""

"""
LlavaVideo_utils.py

LlavaVideo 모델을 사용하는데 필요한 함수들을 적어두었습니다.
바깥 코드에서는 load_llava_video_model, get_video_and_input_ids 함수만 사용합니다.
저 2개의 코드들만 보셔도 됩니다.

함수 목록:
1. load_video
2. load_llava_video_model
3. get_video_and_input_ids
"""

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


class DescriptionGenerator:
    def __init__(self, model_path, input_size=448):
        self.input_size = input_size
        self.model_path = model_path

        self.generation_config = dict(do_sample=False)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = (
            AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            .eval()
            .cuda()
        )

    def _build_transform(self):
        """
        이미지 혹은 분할된 타일 이미지를 텐서로 변환할 때 사용할 transform을 정의합니다.
        """
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (self.input_size, self.input_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        return transform

    def _get_index(self, bound, fps, max_frame, first_idx=0, num_segments=32):
        """
        비디오에서 프레임 인덱스를 추출합니다.
        bound: (start_second, end_second)의 튜플
        """
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array(
            [
                int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
                for idx in range(num_segments)
            ]
        )
        return frame_indices

    def _find_closest_aspect_ratio(
        self, aspect_ratio, target_ratios, width, height, input_size
    ):
        """
        가능한 target_ratios 중에서 현재 이미지의 종횡비(aspect_ratio)에 가장 가까운 것을 찾습니다.
        """
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height

        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                # 같은 차이라면 좀 더 큰 면적을 커버할 수 있는 비율을 선택하도록 예시 조건
                if area > 0.5 * (input_size**2) * ratio[0] * ratio[1]:
                    best_ratio = ratio

        return best_ratio

    def _dynamic_preprocess(self, image, min_num=4, max_num=12, use_thumbnail=False):
        """
        이미지를 일정 블록 수로 잘라내어 처리합니다.
         - min_num, max_num: 가능한 블록 개수의 범위
         - use_thumbnail: 블록을 여러 개로 자른 경우 원본 썸네일을 추가로 넣을지 여부
        """
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # 가능한 블록 조합 (i, j)를 수집
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        # 블록의 총 개수(=i*j)가 작은 순으로 정렬
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # 가장 가까운 종횡비 선택
        chosen_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, self.input_size
        )

        # target_width, target_height 계산 (정수 변환)
        target_width = int(self.input_size * chosen_ratio[0])
        target_height = int(self.input_size * chosen_ratio[1])
        blocks = chosen_ratio[0] * chosen_ratio[1]

        # 이미지 리사이즈
        # (PIL에서 resize에 들어가는 인자는 정수여야 하므로 int 변환)
        resized_img = image.resize((target_width, target_height), Image.BICUBIC)

        processed_images = []
        # 각 블록 단위로 crop
        num_blocks_w = target_width // self.input_size
        num_blocks_h = target_height // self.input_size
        for i in range(int(blocks)):
            box = (
                (i % num_blocks_w) * self.input_size,
                (i // num_blocks_w) * self.input_size,
                ((i % num_blocks_w) + 1) * self.input_size,
                ((i // num_blocks_w) + 1) * self.input_size,
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        # 블록이 여러 개일 때 썸네일(원본 크기 유지 아님. input_size로 리사이즈)을 추가하고 싶다면
        # use_thumbnail=True로 설정
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize(
                (self.input_size, self.input_size), Image.BICUBIC
            )
            processed_images.append(thumbnail_img)

        return processed_images

    def load_video(self, video_path, bound=None, max_num=1, num_segments=32):
        """
        비디오를 읽어들여 지정된 개수(num_segments)의 프레임을 뽑고,
        각 프레임을 _dynamic_preprocess로 분할/처리한 뒤 텐서화하여 합칩니다.
        """
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list = []
        num_patches_list = []
        transform = self._build_transform()

        # 비디오에서 프레임 인덱스 추출
        frame_indices = self._get_index(
            bound, fps, max_frame, first_idx=0, num_segments=num_segments
        )

        for frame_index in frame_indices:
            # decord VideoReader를 통해 frame_index번째 프레임을 읽어 PIL 이미지로 변환
            img = Image.fromarray(vr[frame_index].numpy()).convert("RGB")

            # _dynamic_preprocess로 분할 처리
            img_tiles = self._dynamic_preprocess(
                img, use_thumbnail=True, max_num=max_num
            )

            # 각 타일을 텐서화 및 정규화
            tile_tensors = [transform(tile) for tile in img_tiles]
            tile_tensors = torch.stack(tile_tensors)

            num_patches_list.append(tile_tensors.shape[0])
            pixel_values_list.append(tile_tensors)

        # 모든 프레임의 타일 텐서들을 이어 붙임
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    def load_image(self, image_path, max_num=12):
        """
        이미지를 읽어들여 _dynamic_preprocess로 타일 분할 후 텐서화.
        """
        image = Image.open(image_path).convert("RGB")
        transform = self._build_transform()

        # 이미지 분할 처리
        patches = self._dynamic_preprocess(image, max_num=max_num, use_thumbnail=False)

        # 각 타일을 텐서화
        pixel_values = [transform(patch) for patch in patches]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def describe_scene(self, video_path, question, num_segments=32, max_new_tokens=256):
        """
        비디오에 대한 질문을 입력하면, 분할된 프레임 타일 텐서와 함께 모델에게 질의하여 답변을 생성합니다.
        """
        pixel_values, num_patches_list = self.load_video(
            video_path, num_segments=num_segments
        )
        # print(pixel_values.shape) # 비디오(Scene) 1개마다 출력되서 주석 처리함
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        # 영상의 각 프레임을 순서대로 prompt에 추가
        video_prefix = "".join(
            [f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))]
        )
        query = video_prefix + question

        self.generation_config["max_new_tokens"] = max_new_tokens

        # 모델과 대화(chat) 방식으로 질의
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            query,
            self.generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )
        return response

    def describe_frame(self, image_path, question):
        """
        단일 이미지를 모델에게 질의하여 답변을 생성합니다.
        """
        pixel_values = self.load_image(image_path)
        # print(pixel_values.shape) # 사진 1장마다 출력되서 주석 처리함
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        query = f"<image>\n{question}"

        # 모델과 대화(chat) 방식으로 질의
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            query,
            self.generation_config,
            num_patches_list=[pixel_values.shape[0]],
            history=None,
            return_history=False,
        )
        return response


# 예시 실행
if __name__ == "__main__":
    # 예시 모델 경로(수정하여 사용)
    model_path = "OpenGVLab/InternVideo2_5_Chat_8B"
    generator = DescriptionGenerator(model_path=model_path)

    # 비디오 예시
    video_path = "/data/ephemeral/home/yunseo/benchmark_mini/dataset/scenes/KNAgFhh1ji4_scene_0007.mp4"
    video_question = "Describe this video in detail."
    video_response = generator.describe_scene(
        video_path, video_question, num_segments=32
    )
    print("[Video Output]\n", video_response)

    # 이미지 예시
    # image_path = "/data/ephemeral/home/yunseo/benchmark_mini/dataset/frames/KNAgFhh1ji4_frame_0451.jpg"
    # image_question = "Describe this image in detail."
    # image_response = generator.describe_frame(image_path, image_question)
    # print("[Image Output]\n", image_response)
