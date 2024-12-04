import os
from glob import glob
from scipy.io import loadmat
import cv2
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
from typing import Tuple, Union, Optional
from warnings import warn
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

from datasets import standardize_dataset_name


def _calc_size(
    img_w: int,
    img_h: int,
    min_size: int,
    max_size: int,
    base: int = 32
) -> Union[Tuple[int, int], None]:
    """
    This function generates a new size for an image while keeping the aspect ratio. The new size should be within the given range (min_size, max_size).

    Args:
        img_w (int): The width of the image.
        img_h (int): The height of the image.
        min_size (int): The minimum size of the edges of the image.
        max_size (int): The maximum size of the edges of the image.
    """
    assert min_size % base == 0, f"min_size ({min_size}) must be a multiple of {base}"
    if max_size != float("inf"):
        assert max_size % base == 0, f"max_size ({max_size}) must be a multiple of {base} if provided"

    assert min_size <= max_size, f"min_size ({min_size}) must be less than or equal to max_size ({max_size})"

    aspect_ratios = (img_w / img_h, img_h / img_w)
    # possible to resize and preserve the aspect ratio
    if min_size / max_size <= min(aspect_ratios) <= max(aspect_ratios) <= max_size / min_size:
        # already within the range, no need to resize
        if min_size <= min(img_w, img_h) <= max(img_w, img_h) <= max_size:
            ratio = 1.
        elif min(img_w, img_h) < min_size:  # smaller than the minimum size, resize to the minimum size
            ratio = min_size / min(img_w, img_h)
        else:  # larger than the maximum size, resize to the maximum size
            ratio = max_size / max(img_w, img_h)

        new_w, new_h = int(round(img_w * ratio / base) *
                           base), int(round(img_h * ratio / base) * base)
        new_w = max(min_size, min(max_size, new_w))
        new_h = max(min_size, min(max_size, new_h))
        return new_w, new_h

    else:  # impossible to resize and preserve the aspect ratio
        msg = f"Impossible to resize {img_w}x{img_h} image while preserving the aspect ratio to a size within the range ({min_size}, {max_size}). Will not limit the maximum size."
        warn(msg)
        return _calc_size(img_w, img_h, min_size, float("inf"), base)


def _generate_random_indices(
    total_size: int,
    out_dir: str,
) -> None:
    """
    Generate randomly selected indices for labelled data in semi-supervised learning.
    """
    rng = np.random.default_rng(42)
    for percent in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        num_select = int(total_size * percent)
        selected = rng.choice(total_size, num_select, replace=False)
        selected.sort()
        selected = selected.tolist()
        with open(os.path.join(out_dir, f"{int(percent * 100)}%.txt"), "w") as f:
            for i in selected:
                f.write(f"{i}\n")


def _resize(image: np.ndarray, label: np.ndarray, min_size: int, max_size: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    image_h, image_w, _ = image.shape
    new_size = _calc_size(image_w, image_h, min_size, max_size)
    if new_size is None:
        return image, label, False
    else:
        new_w, new_h = new_size
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC) if (
            new_w, new_h) != (image_w, image_h) else image
        label = label * np.array([[new_w / image_w, new_h / image_h]]) if len(
            label) > 0 and (new_w, new_h) != (image_w, image_h) else label
        return image, label, True


def _preprocess(
    annotations_path: str,
    data_images_dir: str,
    data_dst_dir: str,
    min_size: int,
    max_size: int,
    generate_npy: bool = False
) -> None:
    """
    This function organizes the data into the following structure:
    data_dst_dir
    ├── train
    │   ├── images
    │   │   ├── 0001.jpg
    │   │   ├── 0002.jpg
    │   │   ├── ...
    │   │   images_npy
    │   │   ├── 0001.npy
    │   │   ├── 0002.npy
    │   │   ├── ...
    │   ├── labels
    │   │   ├── 0001.npy
    │   │   ├── 0002.npy
    │   │   ├── ...
    │   ├── 0.01%.txt
    │   ├── 0.05%.txt
    │   ├── ...
    ├── val
    │   ├── images
    │   │   ├── 0001.jpg
    │   │   ├── 0002.jpg
    │   │   ├── ...
    │   │   images_npy
    │   │   ├── 0001.npy
    │   │   ├── 0002.npy
    │   │   ├── ...
    │   ├── labels
    │   │   ├── 0001.npy
    │   │   ├── 0002.npy
    │   │   ├── ...
    """

    os.makedirs(data_dst_dir, exist_ok=True)
    _viso(annotations_path, data_images_dir,
          data_dst_dir, min_size, max_size, generate_npy)


def _resize_and_save(
    image: np.ndarray,
    name: str,
    image_dst_dir: str,
    generate_npy: bool,
    label: Optional[np.ndarray] = None,
    label_dst_dir: Optional[str] = None,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
) -> None:
    os.makedirs(image_dst_dir, exist_ok=True)

    if label is not None:
        assert label_dst_dir is not None, "label_dst_dir must be provided if label is provided"
        os.makedirs(label_dst_dir, exist_ok=True)

    image_dst_path = os.path.join(image_dst_dir, f"{name}.jpg")

    if label is not None:
        label_dst_path = os.path.join(label_dst_dir, f"{name}.npy")
    else:
        label = np.array([])
        label_dst_path = None

    if min_size is not None:
        assert max_size is not None, f"max_size must be provided if min_size is provided, got {max_size}"
        image, label, success = _resize(image, label, min_size, max_size)
        if not success:
            print(f"image: {image_dst_path} is not resized")

    cv2.imwrite(image_dst_path, image)

    if label_dst_path is not None:
        np.save(label_dst_path, label)

    if generate_npy:
        image_npy_dst_path = os.path.join(image_dst_dir, f"{name}.npy")
        image_npy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB
        image_npy = np.transpose(image_npy, (2, 0, 1))  # HWC to CHW
        # Don't normalize the image. Keep it as np.uint8 to save space.
        # image_npy = image_npy.astype(np.float32) / 255.  # normalize to [0, 1]
        np.save(image_npy_dst_path, image_npy)


def parse_cvat_xml(file_path):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Initialize lists
    image_list = []
    label_list = []

    # Iterate through each image in the XML
    for image in root.findall('image'):
        # Get the filename
        filename = image.get('name')
        image_list.append(filename)

        # Collect points for this image
        points_list = []
        for point in image.findall('points'):
            # Get the points attribute and convert to a tuple of floats
            points = point.get('points')
            if ";" in points:
                continue
            points_list.append(list(map(float, points.split(','))))

        label_list.append(points_list)

    return image_list, label_list


def _viso(
    cvat_annotations_path: str,
    images_src_dir: str,
    data_dst_dir: str,
    min_size: int,
    max_size: int,
    generate_npy: bool = False
) -> None:

    image_list, label_list = parse_cvat_xml(cvat_annotations_path)
    image_list_train, image_list_val, label_list_train, label_list_val = train_test_split(
        image_list, label_list, test_size=0.2, random_state=42)

    for split in ["train", "val"]:
        image_dst_dir = os.path.join(data_dst_dir, split, "images")
        label_dst_dir = os.path.join(data_dst_dir, split, "labels")
        os.makedirs(image_dst_dir, exist_ok=True)
        os.makedirs(label_dst_dir, exist_ok=True)

        if split == "train":
            image_list_split, label_list_split = image_list_train, label_list_train
        else:
            image_list_split, label_list_split = image_list_val, label_list_val

        for i in tqdm(range(len(image_list_split))):
            image = cv2.imread(os.path.join(
                images_src_dir, image_list_split[i]))

            label = label_list_split[i]
            # name = os.path.splitext(image_list_split[i])[0]
            name = f"{i+1:04}"

            _resize_and_save(
                image=image,
                label=label,
                name=name,
                image_dst_dir=image_dst_dir,
                label_dst_dir=label_dst_dir,
                generate_npy=generate_npy,
                min_size=min_size,
                max_size=max_size
            )

    # for split in ["train", "val"]:
    #     generate_npy = generate_npy and split == "train"
    #     print(f"Processing {split}...")
    #     if split == "train":
    #         image_src_dir = os.path.join(data_src_dir, "train_data", "images")
    #         label_src_dir = os.path.join(
    #             data_src_dir, "train_data", "ground-truth")
    #         image_src_paths = glob(os.path.join(image_src_dir, "*.jpg"))
    #         label_src_paths = glob(os.path.join(label_src_dir, "*.mat"))
    #         assert len(image_src_paths) == len(label_src_paths) in [
    #             300, 400], f"Expected 300 (part_A) or 400 (part_B) images and labels, got {len(image_src_paths)} images and {len(label_src_paths)} labels"
    #     else:
    #         image_src_dir = os.path.join(data_src_dir, "test_data", "images")
    #         label_src_dir = os.path.join(
    #             data_src_dir, "test_data", "ground-truth")
    #         image_src_paths = glob(os.path.join(image_src_dir, "*.jpg"))
    #         label_src_paths = glob(os.path.join(label_src_dir, "*.mat"))
    #         assert len(image_src_paths) == len(label_src_paths) in [
    #             182, 316], f"Expected 182 (part_A) or 316 (part_B) images and labels, got {len(image_src_paths)} images and {len(label_src_paths)} labels"

    #     def sort_key(x): return int(
    #         (os.path.basename(x).split(".")[0]).split("_")[-1])
    #     image_src_paths.sort(key=sort_key)
    #     label_src_paths.sort(key=sort_key)

    #     image_dst_dir = os.path.join(data_dst_dir, split, "images")
    #     label_dst_dir = os.path.join(data_dst_dir, split, "labels")
    #     os.makedirs(image_dst_dir, exist_ok=True)
    #     os.makedirs(label_dst_dir, exist_ok=True)

    #     size = len(str(len(image_src_paths)))
    #     for i, (image_src_path, label_src_path) in tqdm(enumerate(zip(image_src_paths, label_src_paths)), total=len(image_src_paths)):
    #         image_id = int(
    #             (os.path.basename(image_src_path).split(".")[0]).split("_")[-1])
    #         label_id = int(
    #             (os.path.basename(label_src_path).split(".")[0]).split("_")[-1])
    #         assert image_id == label_id, f"Expected image id {image_id} to match label id {label_id}"
    #         name = f"{(i + 1):0{size}d}"
    #         image = cv2.imread(image_src_path)
    #         label = loadmat(label_src_path)["image_info"][0][0][0][0][0]
    #         # label = [[ 29.6225116  472.92022152]
    #         #  [ 54.35533603 454.96602305]
    #         #  [ 51.79045053 460.46220626]
    #         #  ...
    #         #  [597.89732076 688.27900015]
    #         #  [965.77518336 638.44693908]
    #         #  [166.9965574  628.1873971 ]]
    #         _resize_and_save(
    #             image=image,
    #             label=label,
    #             name=name,
    #             image_dst_dir=image_dst_dir,
    #             label_dst_dir=label_dst_dir,
    #             generate_npy=generate_npy,
    #             min_size=min_size,
    #             max_size=max_size
    #         )

    #     if split == "train":
    #         _generate_random_indices(
    #             len(image_src_paths), os.path.join(data_dst_dir, split))


def parse_args():
    parser = ArgumentParser(
        description="Pre-process datasets to resize images and labeld into a given range.")
    parser.add_argument("--annotations_path", type=str, required=True,
                        help="The path of the CVAT annotation file.")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="The root directory of the source dataset.")
    parser.add_argument("--dst_dir", type=str, required=True,
                        help="The root directory of the destination dataset.")
    parser.add_argument("--min_size", type=int, default=256,
                        help="The minimum size of the shorter side of the image.")
    parser.add_argument("--max_size", type=int, default=None,
                        help="The maximum size of the longer side of the image.")
    parser.add_argument("--generate_npy", action="store_true",
                        help="Generate .npy files for images.")

    args = parser.parse_args()
    args.annotations_path = os.path.abspath(args.annotations_path)
    args.images_dir = os.path.abspath(args.images_dir)
    args.dst_dir = os.path.abspath(args.dst_dir)
    args.max_size = float("inf") if args.max_size is None else args.max_size
    return args


if __name__ == "__main__":
    args = parse_args()
    _preprocess(
        annotations_path=args.annotations_path,
        data_images_dir=args.images_dir,
        data_dst_dir=args.dst_dir,
        min_size=args.min_size,
        max_size=args.max_size,
        generate_npy=args.generate_npy
    )
