import cv2
from PIL import Image
import numpy as np
import os
import logging
from scipy.fftpack import dct, idct

logging.basicConfig(level="INFO")


class HashFunction:
    """
    Hash function class to encode image into hexadecimal hash
    """

    def __init__(self, dir_path: str):
        self.img_hash_dic = {}
        self.dir_path = dir_path

    @staticmethod
    def dct_2d(img):
        return dct(dct(img.T, norm="ortho").T, norm="ortho")

    def phash(self, image_path: str):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_dct = self.dct_2d(gray_img)
        img_idct = dct(dct(img_dct.T, norm="ortho").T, norm="ortho")
        avg_dct = np.mean(img_dct)
        bits = np.asarray(img_dct) > avg_dct
        bit_string = "".join(str(bit) for bit in 1 * bits.flatten())
        hexadecimal_hash = hex(int(bit_string, 2))
        return hexadecimal_hash

    def img_hash_generator(self):
        """
        List directory files and hash each of the files
        """
        f_list = os.listdir(self.dir_path)
        for file in f_list:
            file_path = os.path.join(self.dir_path, file)
            file_hash = self.phash(file_path)
            yield file, file_hash

    def check_duplicates(self):
        """
        Check duplicates and store hashes in a dictionary
        """
        for file, file_hash in self.img_hash_generator():
            if file_hash in self.img_hash_dic:
                print(f"{file} is duplicated!")
            else:
                self.img_hash_dic[file_hash] = file


if __name__ == "__main__":
    IMG_PATH = "/Users/maixueqiao/Downloads/project/makthemak/public/avatar.png"
    IMG1_PATH = "/Users/maixueqiao/Downloads/project/makthemak/public/avatar.png"
    DIR_PATH = "../sample_img"
    hash_fn = HashFunction(dir_path=DIR_PATH)
    hash_fn.check_duplicates()
