# Rank Fusion for Multi-modal Retrieval

This project combines results from two different retrieval methods, **BGE Retrieval** (text-based) and **CLIP Retrieval** (image-based), to produce a fused ranking using weighted scores.

## Overview

The `fuse_results` function takes the outputs of BGE and CLIP retrieval methods and combines them into a single ranked list based on a weighted score. This approach leverages the strengths of both retrieval methods to improve overall result relevance.

## Features

- **BGE Retrieval**: Retrieves text-based results based on a user query.
- **CLIP Retrieval**: Matches text queries with image results.
- **Rank Fusion**: Combines results using weighted scores from both retrieval methods.

## Example Use Case

The example demonstrates:
1. Creating BGE and CLIP retrieval objects.
2. Performing retrieval using a user query.
3. Fusing the results with customizable weights.

## Requirements

- Python 3.x
- YAML configuration files for BGE and CLIP retrieval (`config/basic_config.yaml`, `config/clip_config.yaml`)

Install necessary libraries (if required):
```bash
pip install -r requirements.txt
```

## How to Use

### 1. Prepare Configuration Files

Ensure you have the appropriate YAML configuration files for both BGE and CLIP retrieval methods:
- `config/basic_config.yaml`
- `config/clip_config.yaml`

### 2. Function Parameters

#### `fuse_results` Function

| Parameter      | Description                                       |
|----------------|---------------------------------------------------|
| `text_results` | List of results from BGE Retrieval               |
| `image_results`| List of results from CLIP Retrieval              |
| `w_bge`        | Weight for BGE scores (default: 0.5)             |
| `w_clip`       | Weight for CLIP scores (default: 0.5)            |
| `top_k`        | Number of top results to return (default: 10)    |

Each result list should follow the structure:

**Text Results Example**:
```python
[
  {
    'rank': 1,
    'frame_timestamp': '50.5',
    'frame_image_path': './test_dataset_79/5qlG1ODkRWw_50.500.jpg',
    'score': 0.8501
  },
  ...
]
```

**Image Results Example**:
```python
[
  {
    'rank': 1,
    'image_filename': 'test_dataset_79/mDUSjBiHYeY_29.750.jpg',
    'score': 0.2800
  },
  ...
]
```

### 3. Execute the Script

Run the script:
```bash
python rank_fusion.py
```

### 4. Adjust Weights

Customize the weights for BGE and CLIP scores:
```python
fused_top_k = fuse_results(
    bge_results, clip_results, w_bge=0.7, w_clip=0.3, top_k=5
)
```

### 5. View Results

The fused results will be printed:
```text
=== Rank Fusion Results ===
Rank 1: filename= 5qlG1ODkRWw_50.500.jpg, FinalScore=0.6100, BGE=0.8501, CLIP=0.2800
...
```

## Implementation Details

### Key Steps in `fuse_results`

1. **Dictionary Conversion**: Convert the input lists (`text_results` and `image_results`) into a dictionary for efficient processing. Each key corresponds to a unique filename.
2. **Score Fusion**: Compute the weighted average of BGE and CLIP scores.
3. **Ranking**: Sort the results by the final fused score in descending order.

### Example Workflow in `__main__`

1. Initialize BGE and CLIP retrievers with their respective configuration files.
2. Retrieve results for a sample query (`user_query`).
3. Use `fuse_results` to combine the top results from both methods.
4. Display the fused results.

## Customization

- **Top-k Results**: Adjust the number of top results using the `top_k` parameter in the `fuse_results` function.
- **Weight Adjustment**: Modify `w_bge` and `w_clip` to prioritize one retrieval method over the other.

## Example Output

For a query like "monkey hitting man," the fused results might look like:
```text
=== Rank Fusion Results ===
Rank 1: filename= test_dataset_79/mDUSjBiHYeY_29.750.jpg, FinalScore=0.6500, BGE=0.4000, CLIP=0.9000
Rank 2: filename= test_dataset_79/5qlG1ODkRWw_50.500.jpg, FinalScore=0.6100, BGE=0.8501, CLIP=0.2800
...
```

## License

This project is licensed under the MIT License.

---

Feel free to customize the script to meet your specific use case!
