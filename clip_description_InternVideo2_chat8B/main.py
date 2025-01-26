import os
import json
import yaml
from tqdm import tqdm
from modules.model_utils import *
from modules.video_processing import *
from googletrans import Translator
import torch

#import os 
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

def main():
    # Load configuration from YAML file
    with open("config/config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    
    
    video_dir = config["video_dir"]
    output_directory = config["clip_output_directory"] #full_clip저장할곳
    timestamp_file = config["timestamp_file"]
    folder_name = config["output_folder"] #script
    model_path = config["model_path"] 
    #max_num_frames = config["max_num_frames"]
    generation_config = config["generation_config"]
    prompt = config["prompt"]
    final_output_path = config["final_output"]
    
    # Output folder setup
    os.makedirs(folder_name, exist_ok=True)

    # Get video files
    video_files = [file for file in os.listdir(video_dir) if file.endswith(".mp4")]
    print(video_files)
    save_timestamps_to_txt(video_dir, timestamp_file, threshold=30.0, min_scene_len=2)
    # 동영상 클립 분할 실행
    split_clips_from_txt(video_dir, output_directory, timestamp_file)
    #5개만 우선 테스트해봅니다.
    #video_files = video_files[:2]    
    
    # Initialize model and tokenizer
    model, tokenizer = initialize_model(model_path)
    
    # Initialize Google Translator
    translator = Translator()
    
    # 최종 JSON 데이터 구조 생성
    final_json_data = {"video_clips_info": []}
    
    # Process video and save results
    for video_file in tqdm(video_files):
        print(f"Processing video: {video_file}")
        # 비디오 경로 및 출력 파일 경로 설정
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]  # 비디오 이름 추출
        output_json_path = f"{folder_name}/output_segments_script_{video_name}.json"
        
        # 동영상 처리
        process_video(video_path, output_json_path,timestamp_file)
        
        #with open(output_json_path, 'r') as f:
        #    scripts = json.load(f)

        scene_timestamps = read_timestamps_from_txt(timestamp_file)
        # 각 장면별 클립 생성
        for idx, (start, end) in enumerate(scene_timestamps[video_name]):
            print(f"Processing clip {idx+1}/{len(scene_timestamps[video_name])} for {video_name}")
            clip_name = f"{os.path.splitext(video_file)[0]}_clip_{start:.3f}-{end:.3f}.mp4"
            clip_id = clip_name[:-4]
            clip_path = os.path.join(output_directory, clip_name)

            with open(output_json_path, 'r') as f:
                scripts = json.load(f)        
            
            script_texts = {}
            for script in scripts:
                script_texts[f"{video_name}_clip_{script['start']:.3f}-{script['end']:.3f}"] = script_texts.get(f"{video_name}_clip_{script['start']:.3f}-{script['end']:.3f}","") + f" \n [script]: {script['text']}\n"    
            
            #비디오 클립 로드 및 모델 처리
            video_tensor = load_video(clip_path, num_segments=8, return_msg=False)
            video_tensor = video_tensor.to(model.device)

            if clip_id in script_texts:
                chat_history= []
                response, chat_history = model.chat(tokenizer, '', prompt['clip_prompt_template'], media_type='video', media_tensor=video_tensor, chat_history= chat_history, return_history=True,generation_config=generation_config)
                #print("==============")
                #print(f"Generated response: {response}")
                
                translated_description = translator.translate(response, src='en',dest='ko').text
                #print(f"Translated response: {translated_description}")
                #print("==============")
                for script in scripts:
                    if f"{video_name}_clip_{script['start']:.3f}-{script['end']:.3f}" == clip_id:
                        final_json_data["video_clips_info"].append({
                            "video_id": video_name,
                            "start_time": start,
                            "end_time": end,
                            "clip_id":script["clip"],
                            "clip_description": response,
                            "clip_description_ko": translated_description,
                            "script": script["text"],
                        })
                    #print(f"Current JSON data: {json.dumps(final_json_data, indent=4, ensure_ascii=False)}")
                

                # mp4 지우기 
                #if video_file.endswith(".mp4"):
                #    vidoe_id = video_file[:-4]
                
                        
    # 결과를 json파일로 저장
    sorted_clips_info  = sorted(final_json_data['video_clips_info'],key=lambda x: (x['video_id'], x['start_time']),reverse=False)
    final_json_data['video_clips_info'] = sorted_clips_info

    with open(final_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(final_json_data, json_file, ensure_ascii=False, indent=4)
    
    print(f"All outputs have been saved to {final_output_path}.")


if __name__=="__main__":
    main()