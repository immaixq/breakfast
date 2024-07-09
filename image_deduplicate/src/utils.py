from PIL import Image
from pillow_heif import register_heif_opener
import os
import logging
import argparse

register_heif_opener()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def convert_img_format(img_dir, output_format="jpg"):
    """
    Convert all the images in the specified directory to a different format.

    Args:
        img_dir (str): The path to the directory containing the images.
        output_format (str, optional): The format to convert the images to. Defaults to "jpg".
    """

    images = [
        f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))
    ]
    output_dir = os.path.join(img_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    for image in images:
        logger.info(f"Converting {image}")
        img = Image.open(os.path.join(img_dir, image))
        img.convert("RGB").save(
            os.path.join(output_dir, os.path.splitext(image)[0] + "." + output_format)
        )


def crop_img(img_path, output_dir, resized_w=640, resized_h=640, crop_w=0, crop_h=0):
    """
    Crop an image to a desired size and save the cropped image to a specified output directory.

    Args:
        img_path (str): The path to the input image.
        output_dir (str): The directory where the cropped image will be saved.
        resized_w (int, optional): The desired width of the cropped image. Defaults to 640.
        resized_h (int, optional): The desired height of the cropped image. Defaults to 640.
        crop_w (int, optional): The width of the cropped area. Defaults to 0.
        crop_h (int, optional): The height of the cropped area. Defaults to 0.

    """

    img = Image.open(img_path)
    w, h = img.size
    crop_w = min(w, h)
    left, top = (w - crop_w) // 2, (h - crop_h) // 2
    right, bottom = left + crop_w, top + crop_h
    cropped_img = img.crop((left, top, right, bottom))
    resized_img = cropped_img.resize((resized_w, resized_h), Image.LANCZOS)
    resized_img.save(os.path.join(output_dir, "resized_" + os.path.basename(img_path)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str)
    parser.add_argument("--output_format")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--resized_w", type=int, default=640)
    parser.add_argument("--resized_h", type=int, default=640)
    parser.add_argument("--crop_w", type=int, default=0)
    parser.add_argument("--crop_h", type=int, default=0)

    args = parser.parse_args()
    if args.output_format:
        convert_img_format(args.dir_path, args.output_format)

    if args.output_dir:
        for img in os.listdir(args.dir_path):
            img_path = os.path.join(args.dir_path, img)
            crop_img(
                img_path,
                args.output_dir,
                args.resized_w,
                args.resized_h,
                args.crop_w,
                args.crop_h,
            )
