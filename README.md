# TartanAir to NerfStudio Converter
This Python script provides a streamlined way to convert datasets in the TartanAir format into the NerfStudio format. The converter handles image files, optional depth maps, and pose data, ensuring compatibility with NerfStudio requirements.

## Features

- Converts pose data from TartanAir format to NerfStudio's 4x4 transformation matrices.
- Processes RGB images and optional depth maps.
- Automatically saves the converted dataset with a `transforms.json` file and organized directories for images and depth maps.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/IQisMySenpai/tartanAirToNerfstudio.git
    cd tartanAirToNerfstudio
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
- `-p`, `--pose_path` *(required)*: Path to the pose file.
- `-i`, `--images` *(required)*: Path to the folder containing RGB images.
- `-d`, `--depth_images` *(optional)*: Path to the folder containing depth images.
- `-o`, `--output_path` *(optional)*: Path to the output folder. Defaults to `./output`.

### Example

Convert a TartanAir dataset with images and depth maps:

```bash
python tartanair_to_nerfstudio.py -p ~Downloads/P002/pose_left.txt -i ~Downloads/P002/image_left -d ~Downloads/P002/depth_left -o nerfP002
```

Convert a TartanAir dataset with only images:

```bash
python tartanair_to_nerfstudio.py -p ~Downloads/P002/pose_left.txt -i ~Downloads/P002/image_left -o nerfP002
```

### Note
In case you have a dataset with different camera intrinsics as the default tartanair datasets.
You can change the `camera_intrisincs` variable in the `tartanair_to_nerfstudio.py` file to match your dataset.

## Output Structure

After conversion, the output folder will have the following structure:

```
output/
├── transforms.json     # Camera intrinsics and frame data
├── images/             # RGB images
│   ├── 0.png
│   ├── 1.png
│   └── ...
└── depth/              # Depth maps (if provided)
    ├── 0.npy
    ├── 1.npy
    └── ...
```

## How It Works

1. **Pose Conversion**: Converts TartanAir pose data (translation + quaternion) into 4x4 transformation matrices.
2. **Dataset Organization**: Copies images and depth maps (if available) into structured directories.
3. **Transforms File**: Generates a `transforms.json` containing camera intrinsics and metadata for each frame.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Contributing

Feel free to open issues or submit pull requests to improve this project. Contributions are welcome!
