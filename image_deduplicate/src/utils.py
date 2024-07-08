from PIL import Image
from pillow_heif import register_heif_opener
import os
import argparse

register_heif_opener()


def convert_img_format(img_dir, output_format="jpg"):
    """
    Convert image format
    """
    images = [
        f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))
    ]

    # Create output dir
    output_dir = os.path.join(img_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    for image in images:
        print(f"Converting {image}")
        img = Image.open(os.path.join(img_dir, image))
        img.convert("RGB").save(
            os.path.join(output_dir, os.path.splitext(image)[0] + "." + output_format)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str)
    parser.add_argument("--output_format")
    args = parser.parse_args()

    convert_img_format(args.dir_path)
