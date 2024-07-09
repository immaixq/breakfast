import argparse
import os
import tqdm as tqdm
import logging
from hash_function import HashFunction
from heapq import heapify, heappush, heappop

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

    def __init__(self, dir_path: str, remove_dup: bool, remove_n_similar_img: int):
        self.hasher = HashFunction()
        self.img_hash_dic = {}
        self.similar_imgs = {}
        self.similar_heap = []
        self.dir_path = dir_path
        self.remove_dup = remove_dup
        self.remove_n_similar_img = remove_n_similar_img
        self.min_hamming_dist = 80

    def img_hash_generator(self, dir_path):
        """
        List directory files and hash each o
        f the files
        """
        f_list = os.listdir(dir_path)
        for file in tqdm(f_list):
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
        top_n = 200
        hash_d_item = self.generate_next_item_from_hash_d(hash_d)
        for k, v in hash_d_item:
            logger.debug(f"FILENAME FROM DIC: {v}, CURRENT FILENAME: {file}")
            hamming_dist = self.hasher.hamming_distance(k, file_hash)

            if hamming_dist < 80 and hamming_dist != 0:

                if (
                    len(self.similar_heap) < top_n
                    or hamming_dist < self.similar_heap[0][0]
                ):
                    if len(self.similar_heap) == k:
                        heappop(self.similar_heap)
                    heappush(self.similar_heap, (hamming_dist, v))

                if file in self.similar_imgs:
                    self.similar_imgs[file].append(v)
                else:
                    self.similar_imgs[file] = [v]

        return self.similar_imgs, [v for _, v in self.similar_heap]

    def img_hash_generator(self):
        """
        List directory files and hash each of the files
        """
        f_list = os.listdir(self.dir_path)
        for file in f_list:
            file_path = os.path.join(self.dir_path, file)
            file_hash = self.hasher.phash(file_path)
            yield file, file_hash

    def check_duplicates(self):
        """
        Check duplicates and store hashes in a dictionary
        """
        self.similar_imgs_d = {}
        duplicated_d = {}
        for file, file_hash in self.img_hash_generator():
            if file_hash in self.img_hash_dic:
                if file_hash in duplicated_d:
                    duplicated_d[file_hash].append(file)
                else:
                    duplicated_d[file_hash] = [file]
            else:
                self.img_hash_dic[file_hash] = file

            # get cloest hamming dist
            self.similar_imgs, heap = self.get_closest_hamming_distance(
                file, file_hash, self.img_hash_dic
            )

        return duplicated_d, heap

    def execute(self):
        duplicated_d, similar_heap = self.check_duplicates()
        logger.info(f"Similar image heap: {similar_heap}")
        logger.info(
            f"Duplicated count: {len(duplicated_d.values())}, duplicated pairs: {duplicated_d}"
        )
        if self.remove_dup:
            for _, v in duplicated_d.items():
                for idx, _ in enumerate(v):
                    logger.info(
                        f"Removing duplicates: {os.path.join(self.dir_path, v[idx])}"
                    )
                    os.remove(os.path.join(self.dir_path, v[idx]))
                    logger.info(
                        f"Remaining images count: {len(os.listdir(self.dir_path))}"
                    )

        if self.remove_n_similar_img:
            for idx, x in enumerate(similar_heap):
                logger.info(
                    f"Removing similar images: {os.path.join(self.dir_path, x)}"
                )
                try:
                    os.remove(os.path.join(self.dir_path, x))
                    logger.info(
                        f"Remaining images count: {len(os.listdir(self.dir_path))}"
                    )
                except:
                    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str)
    parser.add_argument("--remove_dup", type=bool, default=False)
    parser.add_argument("--remove_n_similar_img", type=bool, default=False)

    args = parser.parse_args()

    img_deduplicator = ImageDeduplicator(
        args.dir_path, args.remove_dup, args.remove_n_similar_img
    )
    img_deduplicator.execute()
