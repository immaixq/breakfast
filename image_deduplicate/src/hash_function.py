import cv2
import numpy as np
import logging
from scipy.fftpack import dct, idct


# Create a custom logger
logger = logging.getLogger(__name__)
# Set the log level
logger.setLevel(logging.DEBUG)
# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# Create a log formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class HashFunction:
    """
    Hash function class to encode image into hexadecimal hash
    """

    @staticmethod
    def dct_2d(img):
        """
        Compute the two-dimensional discrete cosine transform (DCT) of an image.

        Parameters:
            img (ndarray): The input image.

        Returns:
            ndarray: The DCT of the input image.
        """
        return dct(dct(img.T, norm="ortho").T, norm="ortho")

    def phash(self, image_path: str):
        """
        Hashing function for image
        """
        img = cv2.imread(image_path)
        img = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_dct = self.dct_2d(gray_img)
        img_idct = dct(dct(img_dct.T, norm="ortho").T, norm="ortho")
        avg_dct = np.mean(img_dct)
        bits = np.asarray(img_dct) > avg_dct
        bit_string = "".join(str(bit) for bit in 1 * bits.flatten())
        hexadecimal_hash = hex(int(bit_string, 2))[2:]
        return hexadecimal_hash

    @staticmethod
    def hamming_distance(hash1, hash2):
        """
        Hamming dist to compute differences between two hash
        """
        if len(hash1) != len(hash2):
            raise ValueError("Both hashes must have the same length.")

        # Convert hexadecimal hashes to binary representations
        bin_hash1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
        bin_hash2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)

        # Calculate the Hamming distance
        distance = sum(x1 != x2 for x1, x2 in zip(bin_hash1, bin_hash2))
        return distance
