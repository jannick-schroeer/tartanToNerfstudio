# TartanAir to NerfStudio Converter
TartanToNerfStudio is a Python script that converts the TartanAir and TartanGround dataset format into the NerfStudio dataset format. 
It processes pose files, converts depth images (if needed), and generates a transforms.json file compatible with NerfStudio.

## Features
- Reads pose data from TartanAir datasets.
- Supports three camera intrinsic configurations: Air, Ground, and Custom.
- Converts pose data from TartanAir format to NerfStudio's 4x4 transformation matrices.
- Optionally limits the number of poses and distributes them uniformly.
- Converts depth images to .npy format if required.
- Outputs a transforms.json file compatible with NerfStudio.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/IQisMySenpai/tartanToNerfstudio.git
    cd tartanToNerfstudio
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Command-Line Interface

Run the script with the following arguments:

```bash
python tartanair_to_nerfstudio.py -p <pose_file_path> -i <image_folder_path> [-d <depth_folder_path>] [-o <output_folder_path>]
```

#### Arguments:
| Argument | Description |
|----------|-------------|
| `base_path` | Path to the base folder of the TartanAir dataset (Required). |
| `-c`, `--camera-intrinsics` | Camera intrinsics to use: `Air`, `Ground`, or `Custom` (Default: `Ground`). |
| `-p`, `--pose-limit` | Limit the number of poses to convert (Optional). |
| `-u`, `--uniform` | Distribute selected poses uniformly instead of taking the first `n` poses (Default: `False`). |
| `-d`, `--depth-conversion` | Convert depth images to `.npy` format (Default: `False`). |

### Example
To convert a dataset with a pose limit of 300 images, which are uniformly selected, using `Ground` camera intrinsics and enabling depth conversion:
```sh
python tartan_to_nerfstudio.py ./dataset -c Ground -du -p 300
```

### Note
In case you have a dataset with different camera intrinsics as the default tartanair datasets.
You can change the `camera_intrisincs` variable in the `tartanair_to_nerfstudio.py` file to match your dataset.

## Output
The script will generate a `transforms.json` file in the base folder with the following structure:
```json
{
    "camera_model": "OPENCV",
    "fl_x": 320.0,
    "fl_y": 320.0,
    "cx": 320.0,
    "cy": 240.0,
    "w": 640,
    "h": 480,
    "frames": [
        {
            "file_path": "image0001.png",
            "transform_matrix": [[...]]
        }
    ]
}
```