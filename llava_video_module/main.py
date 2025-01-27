from module.model_utils import *
import os 
import yaml
from googletrans import Translator
from tqdm import tqdm
import json
from transformers import GenerationConfig
from module.video_processing import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")

def main():
    with open("config/config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
        
    video_dir = config["video_dir"]
    clip_output_directory = config['clip_output_directory'] # clips 저장 
    output_folder = config['output_folder'] # output_folder
    pretrained = config['model_path'] # "lmms-lab/LLaVA-Video-7B-Qwen2"
    final_output = config['final_output'] # "clip_output_v20.json"
    model_name = config['model_name'] # "llava_qwen"
    timestamp_file = config['timestamp_file'] # "/workspace/video_timestamps.txt"
    
    
    max_frames_num = config['max_num_frames']
    generation_config = config["generation_config"]
    fps= config['fps']
    
    device = "cuda"
    # ==============================================
    os.makedirs(output_folder, exist_ok=True)
    generation_config_dict = config.get("generation_config", {})
    generation_config = GenerationConfig(**generation_config_dict)
    # 비디오 파일 목록 생성
    video_files = [file for file in os.listdir(video_dir) if file.endswith(".mp4")]
    save_timestamps_to_txt(video_dir, timestamp_file, threshold=30.0, min_scene_len=2)
    split_clips_from_txt(video_dir, clip_output_directory, timestamp_file)
    
    # 모델 초기화
    tokenizer, model, image_processor, _ = initialize_model(pretrained, model_name, device)
    translator = Translator()
    
    final_json_data = {"video_clips_info": []}
    
    # 각 비디오 파일 처리
    for video_file in tqdm(video_files):
        video_path = os.path.join(video_dir, video_file)
        max_frames_num = max_frames_num

        video_name = os.path.splitext(video_file)[0]   
        output_json_path = f"{output_folder}/output_segments_script_{video_name}.json"
        process_video(video_path, output_json_path,timestamp_file)
    
            
            
        scene_timestamps = read_timestamps_from_txt(timestamp_file)
        # 각 장면별 클립 생성
        for idx, (start, end) in enumerate(scene_timestamps[video_name]):
            clip_name = f"{os.path.splitext(video_file)[0]}_clip_{start:.3f}-{end:.3f}.mp4"
            clip_id = clip_name[:-4]
            clip_path = os.path.join(clip_output_directory, clip_name)
            
            prompt = f"Please explain the video in detail by referring to the lines, focusing on the action and motion of the video"
            
            with open(output_json_path, 'r') as f:
                scripts = json.load(f)
            
            script_texts = {}
            for script in scripts:
                script_texts[f"{video_name}_clip_{script['start']:.3f}-{script['end']:.3f}"] = script_texts.get(f"{video_name}_clip_{script['start']:.3f}-{script['end']:.3f}","") + f" \n [script]: {script['text']}\n"    

            # 비디오 클립 로드 및 모델 처리
            video, frame_time, video_time = load_video(clip_path, max_frames_num, fps=fps, force_sample=True)
            video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].to(device).to(torch.float16)
            video = [video]

            # 질문 생성 Describe및 출력
            prompt_question = create_question(prompt,video_time, len(video[0]), frame_time)
            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
            
            if clip_id in script_texts:
                #prompt += script_texts[clip_id]
                
                attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
                
                cont = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images=video,
                    modalities=["video"],
                    generation_config=generation_config,
                )
                text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
                translated_text = translator.translate(text_outputs, src='en', dest='ko').text

                for script in scripts:
                    if f"{video_name}_clip_{script['start']:.3f}-{script['end']:.3f}" == clip_id:
                        final_json_data["video_clips_info"].append({
                            "video_id": video_name,
                            "start_timestamp": start,
                            "end_timestamp": end,
                            "clip_id":script["clip"],
                            "clip_description": text_outputs,
                            "clip_description_ko": translated_text,
                            "script": script["text"],
                        })
    
    # 결과를 JSON 파일로 저장
    with open(final_output, 'w') as json_file:
        json.dump(final_json_data, json_file, indent=4, ensure_ascii=False)
    print(f"Final output saved to {final_output}")

if __name__ == "__main__":
    main()
