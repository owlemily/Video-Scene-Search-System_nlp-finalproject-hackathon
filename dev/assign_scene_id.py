import json

# Load the JSON data for frames and scenes
with open(
    "/data/ephemeral/home/refactor-retrieval/description/frame_output_test_dataset_79_v1.json",
    "r",
) as frames_file:
    frames_data = json.load(frames_file)

with open(
    "/data/ephemeral/home/refactor-retrieval/description/scene_output_v23.json", "r"
) as scenes_file:
    scenes_data = json.load(scenes_file)

frames_info = frames_data["frames"]

# Extract scenes information
scenes_info = scenes_data["scenes"]

# Assign scene_id to each frame
for frame in frames_info:
    video_id = frame["video_id"]
    timestamp = float(frame["timestamp"])

    # Find the matching scene
    for scene in scenes_info:
        if scene["video_id"] == video_id and float(
            scene["start_time"]
        ) <= timestamp < float(scene["end_time"]):
            frame["scene_id"] = scene["scene_id"]
            break

# Save the updated frames data with scene_id
frames_data["frames"] = frames_info

# Save the updated frames data with scene_id
with open("dev/frames_with_scene_id.json", "w", encoding="utf-8") as output_file:
    json.dump(frames_data, output_file, indent=4, ensure_ascii=False)

print("Scene IDs have been assigned and saved to 'frames_with_scene_id.json'")
