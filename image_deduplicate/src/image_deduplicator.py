import argparse
import os
import logging
from hash_function import HashFunction

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


class ImageDeduplicator:
    """
    Image deduplicator class to encode image into hexadecimal hash
    """

    def __init__(self, dir_path: str):
        self.hasher = HashFunction()
        self.img_hash_dic = {}
        self.similar_imgs = {}
        self.dir_path = dir_path

    def img_hash_generator(self, dir_path):
        """
        List directory files and hash each of the files
        """
        f_list = os.listdir(dir_path)
        for file in f_list:
            logger.info(f"Hashing file: {file}")
            file_path = os.path.join(self.dir_path, file)
            file_hash = self.hasher.phash(file_path)
            yield file, file_hash

    def generate_next_item_from_hash_d(self, hash_d):
        if not hash_d:
            return None
        for k, v in hash_d.items():
            yield k, v

    def get_closest_hamming_distance(self, file, file_hash, hash_d):
        """
        Get similar images based on hamming distance threshold
        """

        hash_d_item = self.generate_next_item_from_hash_d(hash_d)
        for k, v in hash_d_item:
            logger.debug(f"FILENAME FROM DIC: {v}, CURRENT FILENAME: {file}")
            hamming_dist = self.hasher.hamming_distance(k, file_hash)

            if hamming_dist < 80 and hamming_dist != 0:
                if file in self.similar_imgs:
                    self.similar_imgs[file].append(v)
                else:
                    self.similar_imgs[file] = [v]

        return self.similar_imgs

    def img_hash_generator(self):
        """
        List directory files and hash each of the files
        """
        f_list = os.listdir(self.dir_path)
        for file in f_list:
            logger.info(f"Hashing file: {file}")
            file_path = os.path.join(self.dir_path, file)
            file_hash = self.hasher.phash(file_path)
            yield file, file_hash

    def check_duplicates(self):
        """
        Check duplicates and store hashes in a dictionary
        """
        self.similar_imgs_d = {}
        duplicated_pairs = []
        for file, file_hash in self.img_hash_generator():
            if file_hash in self.img_hash_dic:
                duplicated_pairs.append((self.img_hash_dic[file_hash], file))
            else:
                self.img_hash_dic[file_hash] = file

            # get cloest hamming dist
            self.get_closest_hamming_distance(file, file_hash, self.img_hash_dic)

        # logger.info(self.img_hash_dic)
        return duplicated_pairs, self.similar_imgs

    def execute(self):
        duplicated_pairs, similar_imgs = self.check_duplicates()
        logger.info(f"Similar image dictionary: {similar_imgs}")
        logger.info(
            f"Duplicated count: {len(duplicated_pairs)}, duplicated pairs: {duplicated_pairs}"
        )


if __name__ == "__main__":
    # DIR_PATH = "../tests/sample_data"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str)
    args = parser.parse_args()

    img_deduplicator = ImageDeduplicator(args.dir_path)
    img_deduplicator.execute()
