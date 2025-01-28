# RankFusion Retrieval Pipeline

This project implements a rank fusion retrieval pipeline that combines multiple retrieval methods (frame, scene, and image retrieval) into a unified scoring mechanism to provide ranked results based on user queries. The pipeline leverages:

1. **BGERetrieval (Basic Geometric Embedding Retrieval)** for text-based frame and scene retrieval.
2. **CLIPRetrieval** for image-based retrieval using CLIP embeddings.

## Project Structure

```
.
|-- README.md                # Project documentation
|-- code
|   |-- __init__.py          # Initialization script for the code module
|   |-- __pycache__          # Cached Python bytecode files
|   |-- basic_retrieval.py   # Implementation of BGERetrieval for frame and scene
|   -- image_retrieval.py    # Implementation of CLIPRetrieval for image retrieval
|-- config
|   |-- clip_config.yaml               # Configuration file for CLIP retrieval
|   |-- frame_description_config.yaml  # Configuration file for frame retrieval
|   -- scene_description_config.yaml   # Configuration file for scene retrieval
|-- datasets
|   |-- frames_22           # Dataset for frame retrieval
|   -- test_dataset_79      # Test dataset for validation
|-- description
|   |-- frame_output_test_dataset_79_v1.json  # Sample frame retrieval outputs (v1)
|   |-- frame_output_test_dataset_79_v2.json  # Sample frame retrieval outputs (v2)
|   |-- frame_output_v3_unsloth_22.json       # Sample frame retrieval outputs (v3)
|   -- scene_output_v22.json                  # Sample scene retrieval outputs
|-- dev
|   -- assign_scene_id.py   # Script for assigning scene IDs to frames
|-- init_dataset
|   |-- download_test_dataset_79.sh  # Script to download test dataset
|   |-- download_video.sh            # Script to download video data
|   |-- only_extract_frames.py       # Script to extract frames from video
|   -- video                         # Directory for raw video files
|-- rankfusion_retrieval_pipeline.py # Main pipeline script for rank fusion
-- requirements.txt                  # Python dependencies
```

## Key Components

### 1. **BGERetrieval (Basic Geometric Embedding Retrieval)**

This module handles retrieval based on textual descriptions of frames and scenes.

- **Frame Retrieval**: Extracts relevant frames based on user queries.
- **Scene Retrieval**: Extracts relevant scenes based on user queries.

### 2. **CLIPRetrieval**

This module performs image-based retrieval using CLIP embeddings to match images to user queries.

### 3. **Rank Fusion Logic**

The `fuse_results` function combines retrieval results from frame, scene, and image retrieval modules using weighted scores:

- **Input**: Results from the three retrieval methods.
- **Weights**: Adjustable parameters for frame, scene, and clip scores.
- **Output**: Unified top-k results ranked by the final score.

## Configuration Files

- `config/frame_description_config.yaml`: Configuration for frame retrieval.
- `config/scene_description_config.yaml`: Configuration for scene retrieval.
- `config/clip_config.yaml`: Configuration for image retrieval.

## Usage

### Prerequisites

Ensure Python 3 is installed along with the required dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Steps to Run

1. **Prepare Datasets**:

   - Use `init_dataset/download_test_dataset_79.sh` to download the test dataset.


   - Use `init_dataset/only_extract_frames.py` to extract frames from video files.

2. **Run the Pipeline**:
   Execute the main script to perform retrieval and rank fusion:

   ```bash
   python rankfusion_retrieval_pipeline.py
   ```

3. **Example Query**:
   Update the user query in the `rankfusion_retrieval_pipeline.py` script (e.g., `user_query = "monkey hitting man"`).

4. **Output**:
   The script will output the top-k fused results, including individual scores from frame, scene, and image retrieval.

### Sample Output

An example of rank fusion results:

```
=== Rank Fusion Results ===
Rank 1: filename=frame_001.jpg, Final=0.8540, frame=0.30, scene=0.40, clip=0.15, scene_id=12
Rank 2: filename=frame_002.jpg, Final=0.7450, frame=0.25, scene=0.35, clip=0.14, scene_id=8
...
```

## File Descriptions

### Code

- `basic_retrieval.py`: Implements the `BGERetrieval` class for text-based retrieval.
- `image_retrieval.py`: Implements the `CLIPRetrieval` class for image-based retrieval.

### Datasets

- `frames_22`: Contains frame data for retrieval.
- `test_dataset_79`: Contains test data for validation.

### Scripts

- `rankfusion_retrieval_pipeline.py`: Main script to execute the rank fusion pipeline.
- `assign_scene_id.py`: Utility to map scene IDs to frames.
- `only_extract_frames.py`: Extract frames from video files for frame-based retrieval.

## Future Work

- Incorporate advanced rank fusion techniques.
- Add support for real-time retrieval pipelines.
- Improve weight optimization for fused scoring.

---

For questions or contributions, please contact the project maintainer.

