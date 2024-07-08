import pytest

from src.hash_function import HashFunction
import numpy as np
import cv2

class TestHashFunction:

    def setup_method(self, method):
        print(f"Setting up {method}")
        self.image_path = "sample_data/cat.jpg"
        self.hash_function = HashFunction()

    def test_phash(self):
        hash = self.hash_function.phash(self.image_path)
        assert hash == "800280d22df22f742a453e113b04db49f35c9884d0ace1ae8c364c2a882a0c02"
    
    def test_hamming_distance(self):
        # Ensure hamming distance of same hash is 0
        hash1 = "800280d22df22f742a453e113b04db49f35c9884d0ace1ae8c364c2a882a0c02"
        hash2 = "800280d22df22f742a453e113b04db49f35c9884d0ace1ae8c364c2a882a0c02"
        assert self.hash_function.hamming_distance(hash1, hash2) == 0
    
    def teardown_method(self, method):
        print(f"Running tearDown for {method}")
