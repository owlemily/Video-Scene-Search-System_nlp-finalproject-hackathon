import json

# Load the JSON data for frames and scenes
with open(
    "/data/ephemeral/home/refactor-retrieval/description/frame_output_v3_unsloth_22.json",
    "r",
) as frames_file:
    frames_data = json.load(frames_file)

with open(
    "/data/ephemeral/home/refactor-retrieval/description/scene_output_v22.json", "r"
) as scenes_file:
    scenes_data = json.load(scenes_file)

# Extract scenes information
scenes_info = scenes_data["video_scenes_info"]

# Assign scene_id to each frame
for frame in frames_data:
    video_id = frame["video_id"]
    timestamp = float(frame["timestamp"])

    # Find the matching scene
    for scene in scenes_info:
        if (
            scene["video_id"] == video_id
            and scene["start_time"] <= timestamp < scene["end_time"]
        ):
            frame["scene_id"] = scene["scene_id"]
            break

# Save the updated frames data with scene_id
with open("dev/frames_with_scene_id.json", "w", encoding="utf-8") as output_file:
    json.dump(frames_data, output_file, indent=4, ensure_ascii=False)

print("Scene IDs have been assigned and saved to 'frames_with_scene_id.json'")
