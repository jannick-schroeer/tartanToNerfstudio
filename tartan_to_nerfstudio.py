import argparse
import json
import os
import shutil
import warnings
from typing import List, Dict, Any
import inquirer
import numpy as np
from tqdm import tqdm


camera_intrisincs = {
    "camera_model": "OPENCV", # Camera model
    "fl_x": 320.0, # focal length x
    "fl_y": 320.0, # focal length y
    "cx": 320.0, # principal point x
    "cy": 240.0, # principal point y
    "w": 640, # image width
    "h": 480 # image height
}

class TartanToNerfStudio:
    def __init__(self,
                 base_path: str=None
                 ):
        """
        Create a new TartanToNerfStudio converter.
        :param base_path: Path to the base folder of the TartanAir/Ground dataset.
        """
        if base_path is None:
            raise ValueError("Please provide the path to the base folder of the TartanAir/Ground dataset.")

        self.base_path = base_path

    def check_poses(self):
        """
        Check which poses are available in the dataset.
        :return: List of available poses.
        """

        poses = os.listdir(self.base_path)
        poses = [p for p in poses if p.startswith("pose") and p.endswith(".txt")]

        pose_info = []

        for pose in poses:
            # Check if pose has image and/or depth images
            name = pose.replace("pose", "").replace(".txt", "")

            has_rgb = os.path.exists(os.path.join(self.base_path, f"image{name}"))
            has_depth = os.path.exists(os.path.join(self.base_path, f"depth{name}"))

            if not has_rgb:
                continue

            pose_info.append({
                "name": name,
                "has_rgb": has_rgb,
                "has_depth": has_depth
            })

        return pose_info

    def create_transforms(self, poses: List[str]):
        """
        Create the transforms.json file for the NerfStudio dataset.
        :param poses: List of poses to include in the dataset.
        :return: None
        """

        loaded_poses = self.load_poses(poses)
        converted_poses = self.convert_poses(loaded_poses)



    def load_poses(self, poses: List[str]):
        loaded_poses = []

        for name in poses:
            with open(os.path.join(self.base_path, f"pose{name}.txt")) as f:
                lines = f.readlines()
                # Separate the values and convert them to float
                camera_poses = [[float(x) for x in line.strip().split()] for line in lines]
                has_depth = os.path.exists(os.path.join(self.base_path, f"depth{name}"))

                loaded_poses.append({
                    'name': name,
                    'camera_poses': camera_poses,
                    "has_depth": has_depth
                })

        return loaded_poses

    def convert_poses(self, poses: List[Dict]):
        for pose in poses:
            pose['nerfstudio_poses'] = self.convert_pose(pose)

        return poses

    def convert_pose(self, pose: Dict):
        nerfstudio_poses = []
        for camera_pose in pose['camera_poses']:
            x_ned, y_ned, z_ned, qx_ned, qy_ned, qz_ned, qw_ned = camera_pose
            transform_ned = self.quaternion_to_transform(x_ned, y_ned, z_ned, qx_ned, qy_ned, qz_ned, qw_ned)
            nerfstudio_poses.append(self.ned_to_opengl(transform_ned))

        return nerfstudio_poses

    @staticmethod
    def quaternion_to_transform(x: float,
                                y: float,
                                z: float,
                                qx: float,
                                qy: float,
                                qz: float,
                                qw: float) -> np.ndarray:
        """
        Converts OpenGL-compatible coordinates and quaternion into a 4x4 transformation matrix.

        :param x: OpenGL X coordinate (translation)
        :param y: OpenGL Y coordinate (translation)
        :param z: OpenGL Z coordinate (translation)
        :param qx: Quaternion i
        :param qy: Quaternion j
        :param qz: Quaternion k
        :param qw: Quaternion scalar (real part)
        :return: 4x4 transformation matrix in OpenGL coordinates.
        """
        # Normalize the quaternion to ensure it's valid
        norm = np.sqrt(qw ** 2 + qx ** 2 + qy ** 2 + qz ** 2)
        qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

        # Compute the rotation matrix from the quaternion
        rotation_matrix = np.array([
            [qw ** 2 + qx ** 2 - qy ** 2 - qz ** 2, 2 * (qx * qy - qw * qz), 2 * (qw * qy + qx * qz)],
            [2 * (qx * qy + qw * qz), qw ** 2 - qx ** 2 + qy ** 2 - qz ** 2, 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qw * qx + qy * qz), qw ** 2 - qx ** 2 - qy ** 2 + qz ** 2]
        ])

        # Construct the 4x4 transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix  # Set the rotation matrix
        transform_matrix[:3, 3] = [x, y, z]  # Set the translation

        return transform_matrix

    @staticmethod
    def ned_to_opengl(transform_ned: np.ndarray) -> np.ndarray:
        """
        Converts a 4x4 transformation matrix from NED to OpenGL coordinates.

        :param transform: 4x4 transformation matrix in NED coordinates.
        :return: 4x4 transformation matrix in OpenGL
        """
        # NED to OpenGL transformation matrix
        translation = np.array([
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        translation_inv = np.linalg.inv(translation)

        transform_opengl = translation @ transform_ned @ translation_inv

        return transform_opengl

    def write_transforms(self, poses: List[Dict]):
        transforms = camera_intrisincs.copy()
        transforms["frames"] = []

        for pose in poses:
            image_folder_name = f"image_{pose['name']}"
            depth_folder_name = f"depth_{pose['name']}"

            images_folder = os.path.join(self.base_path, image_folder_name)
            images = os.listdir(images_folder)
            images.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(x)[0]))))

            if pose['has_depth']:
                depth_folder = os.path.join(self.base_path, depth_folder_name)
                depth_images = os.listdir(depth_folder)
                depth_images.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(x)[0]))))

            for i, nerf_pose in enumerate(pose['nerfstudio_poses']):
                frame = {
                    "file_path": f"{image_folder_name}/{images[i]}",
                    "transform_matrix": pose.tolist()
                }

                if pose['has_depth']:
                    frame["depth_file_path"] = f"{depth_folder_name}/{depth_images[i]}"

                transforms["frames"].append(frame)

        with open(os.path.join(self.base_path, "transforms.json"), "w") as f:
            json.dump(transforms, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TartanAir dataset to NerfStudio dataset.")
    parser.add_argument("-b", "--base_path", type=str, help="Path to the base folder.")
    args = parser.parse_args()

    converter = TartanToNerfStudio(args.base_path)
    poses = converter.check_poses()

    questions = [
        inquirer.Checkbox('poses',
                          message='Select which poses you want to have in the dataset.',
                          choices=[(f"{pose['name'].strip('_')} (rgb{'+d' if pose['has_rgb'] else ''})", pose['name']) for pose in poses],
                          ),
    ]

    converter.create_transforms(inquirer.prompt(questions)['poses'])