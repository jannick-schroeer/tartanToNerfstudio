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

2. (Optional) Create a conda environment (Python 3.8): I recommend using your existing [nerfstudio environment](https://docs.nerf.studio/quickstart/installation.html).
    ```bash
    conda create -n nerfstudio python=3.8
    ```
3. Activate the environment:
    ```bash
    conda activate nerfstudio
    ```

3. Install the required dependencies:
    ```bash
    pip install -e .
    ```

## Usage

### Command-Line Interface

Run the script using tyro CLI:

```bash
tartanToNerfstudio <base_path> [--pose-limit N] [--uniform] [--depth-conversion] [--camera-intrinsics {Air,Ground,Custom}] [--output-path OUTPUT]
```

#### Arguments:

| Argument             | Type      | Required | Description                                                                                   |
|----------------------|-----------|----------|-----------------------------------------------------------------------------------------------|
| `base_path`          | Path      | Yes      | Path to the base folder of the TartanAir dataset.                                             |
| `--pose-limit`, `-p` | int       | No       | Limit the number of poses to convert.                                                         |
| `--uniform`, `-u`    | bool      | No       | Distribute selected poses uniformly instead of taking the first `n` poses (Default: `False`). |
| `--depth-conversion`, `-d` | bool | No       | Convert depth images to `.npy` format (Default: `False`). Needed for Tartan Ground.           |
| `--camera-intrinsics`, `-c` | Air/Ground/Custom | No | Camera intrinsics to use: `Air`, `Ground`, or `Custom` (Default: `Air`).                      |
| `--output-path`, `-o`| Path      | No       | Output transforms file path (default: transforms.json in base folder).                        |

### Example

Convert a TartanAir dataset with images and depth maps:

```sh
tartanToNerfstudio ./dataset --pose-limit 300 --camera-intrinsics Ground --depth-conversion
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

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Contributing

Feel free to open issues or submit pull requests to improve this project. Contributions are welcome!
