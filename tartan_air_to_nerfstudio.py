import argparse
import json
import os
import shutil
import warnings
from typing import List, Dict, Any

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

def quaternion_multiply(q1, q2):
    """
    Multiplies two quaternions.

    :param q1: First quaternion (qw, qx, qy, qz)
    :param q2: Second quaternion (qw, qx, qy, qz)
    :return: Resulting quaternion (qw, qx, qy, qz)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

class TartanAirToNerfstudio:
    def __init__(self,
                 pose_path: str = None,
                 image_path: str = None,
                 depth_path: str = None,
                 output_path: str = None,
                 ):
        """
        Create a new TartanAirToNerfstudio converter.

        :param pose_path: Path to the pose file. The pose file with format: x y z q0 q1 q2 q3
        :param image_path: Path to the image folder. The images should be in .png, .jpg or .jpeg format.
        :param depth_path: Path to the depth image folder. The depth images should be in .npy, .png, .jpg or .jpeg format.
        :param output_path: Path to the output folder. The output folder will contain the NerfStudio dataset.
        """
        if pose_path is None:
            raise ValueError("Please provide the path to the pose file.")

        if image_path is None:
            raise ValueError("Please provide the path to the image.")


        self.has_depth = (depth_path is not None)
        if self.has_depth:
            self.depth_path = depth_path
        else:
            warnings.warn("Warning: No depth path provided. Depth images will not be converted.")

        if output_path is None:
            output_path = os.path.join(os.getcwd(), "output")

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.pose_path = pose_path
        self.image_path = image_path
        self.output_path = output_path

    def run_conversion(self):
        """
        Run the conversion from TartanAir to NerfStudio.
        :return: None
        """
        images = self.get_images()
        poses = self.load_poses()

        if self.has_depth:
            depth_images = self.get_depth_images()
        else:
            depth_images = []

        if len(poses) != len(images):
            raise ValueError(f"The number of poses ({len(poses)}) does not match the number of images ({len(images)}).")

        if self.has_depth and len(poses) != len(depth_images):
            raise ValueError(f"The number of poses ({len(poses)}) does not match the number of depth images ({len(self.depth_path)}).")

        nerfstudio_poses = self.convert_poses(poses)
        # Copy the camera intrinsics as a new dictionary nerfstudio_transforms
        nerfstudio_transforms = camera_intrisincs.copy()
        nerfstudio_transforms["frames"] = []

        # Sort images by their number content
        images.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(x)[0]))))

        if self.has_depth:
            depth_images.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(x)[0]))))

        for i, pose in enumerate(nerfstudio_poses):
            frame = {
                "file_path": f"images/{images[i]}",
                "transform_matrix": pose.tolist()
            }

            if self.has_depth:
                frame["depth_file_path"] = f"depth/{depth_images[i]}"

            nerfstudio_transforms["frames"].append(frame)

        self.save_nerfstudio_dataset(nerfstudio_transforms)

    def save_nerfstudio_dataset(self, nerfstudio_transforms: Dict[str, Any]):
        """
        Save the NerfStudio dataset.
        :param nerfstudio_transforms: Dictionary containing the camera intrinsics and the list of frames.
        :return: None
        """
        with open(os.path.join(self.output_path, "transforms.json"), "w") as transforms:
            json.dump(obj=nerfstudio_transforms, fp=transforms, indent=4)

        # Create images folder
        images_folder = os.path.join(self.output_path, "images")

        if not os.path.exists(images_folder):
            os.makedirs(images_folder)
        else:
            shutil.rmtree(images_folder)
            os.makedirs(images_folder)

        # Copy images to the images folder
        images = os.listdir(self.image_path)
        for image in tqdm(images, desc="Copying images"):
            if image.endswith(".png") or image.endswith(".jpg") or image.endswith(".jpeg"):
                shutil.copyfile(os.path.join(self.image_path, image), os.path.join(images_folder, image))

        # Create depth folder
        if self.has_depth:
            depth_folder = os.path.join(self.output_path, "depth")
            if not os.path.exists(depth_folder):
                os.makedirs(depth_folder)
            else:
                shutil.rmtree(depth_folder)
                os.makedirs(depth_folder)

            # Copy depth images to the depth folder
            depth_images = os.listdir(self.depth_path)
            for depth_image in tqdm(depth_images, desc="Copying depth images"):
                if depth_image.endswith(".npy") or depth_image.endswith(".png") or depth_image.endswith(".jpg") or depth_image.endswith(".jpeg"):
                    shutil.copyfile(os.path.join(self.depth_path, depth_image), os.path.join(depth_folder, depth_image))

        print(f"Dataset saved to {self.output_path}")

    def get_images(self) -> List[str]:
        """
        Get the list of images in the image folder.
        :return: List of images.
        """
        images = os.listdir(self.image_path)
        return [i for i in images if i.endswith((".png", ".jpg", ".jpeg"))]

    def get_depth_images(self) -> List[str]:
        """
        Get the list of depth images in the depth image folder.
        :return:
        """
        depth_images = os.listdir(self.depth_path)
        return [d for d in depth_images if d.endswith((".npy", ".png", ".jpg", ".jpeg"))]

    def load_poses(self) -> List[List[float]]:
        """
        Load the poses from the pose file.
        :return: List of poses. Each pose is a list of 7 values: x, y, z, q0, q1, q2, q3.
        """
        with open(self.pose_path) as f:
            lines = f.readlines()
            # Separate the values and convert them to float
            poses = [[float(x) for x in line.strip().split()] for line in lines]

        return poses

    def convert_poses(self, poses: List[List[float]]) -> List[np.ndarray]:
        """
        Converts the poses from TartanAir format to NerfStudio format.
        :param poses: List of poses in TartanAir format. Each pose is a list of 7 values: x, y, z, q0, q1, q2, q3.
        :return: List of poses in NerfStudio format. Each pose is a 4x4 transformation matrix.
        """
        nerf_studio_poses = []
        for i, pose in enumerate(poses):
            x_ned, y_ned, z_ned, q0_ned, q1_ned, q2_ned, q3_ned = pose
            transform_ned = self.quaternion_to_transform(x_ned, y_ned, z_ned, q0_ned, q1_ned, q2_ned, q3_ned)
            transform_opengl = self.ned_to_opengl(transform_ned)
            nerf_studio_poses.append(transform_opengl)

        return nerf_studio_poses

    @staticmethod
    def quaternion_to_transform(x: float,
                                y: float,
                                z: float,
                                q0: float,
                                q1: float,
                                q2: float,
                                q3: float) -> np.ndarray:
        """
        Converts OpenGL-compatible coordinates and quaternion into a 4x4 transformation matrix.

        :param x: OpenGL X coordinate (translation)
        :param y: OpenGL Y coordinate (translation)
        :param z: OpenGL Z coordinate (translation)
        :param q0: Quaternion scalar (real part)
        :param q1: Quaternion i
        :param q2: Quaternion j
        :param q3: Quaternion k
        :return: 4x4 transformation matrix in OpenGL coordinates.
        """
        # Normalize the quaternion to ensure it's valid
        norm = np.sqrt(q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2)
        q0, q1, q2, q3 = q0 / norm, q1 / norm, q2 / norm, q3 / norm

        # Compute the rotation matrix from the quaternion
        rotation_matrix = np.array([
            [q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2, 2 * (q1 * q2 - q0 * q3), 2 * (q0 * q2 + q1 * q3)],
            [2 * (q1 * q2 + q0 * q3), q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2, 2 * (q2 * q3 - q0 * q1)],
            [2 * (q1 * q3 - q0 * q2), 2 * (q0 * q1 + q2 * q3), q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2]
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
        translation = np.array([
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1]
        ])

        return np.dot(translation, transform_ned)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TartanAir dataset to NerfStudio dataset.")
    parser.add_argument("-p", "--pose_path", type=str, help="Path to the pose file.")
    parser.add_argument("-i", "--images", type=str, help="Path to the images.")
    parser.add_argument("-d", "--depth_images", type=str, help="Path to the depth images.")
    parser.add_argument("-o", "--output_path", type=str, help="Path to the output folder.", default=os.path.join(os.getcwd(), "output"))

    args = parser.parse_args()

    converter = TartanAirToNerfstudio(args.pose_path, args.images, args.depth_images, args.output_path)
    converter.run_conversion()
