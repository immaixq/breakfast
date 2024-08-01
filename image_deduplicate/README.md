## Introduction 
This python package can be used to find duplicated images and similar images. The package also includes common utils functions such as converting image format for image preparation.

The package uses perceptual hashing, a technique that converts images into compact binary representations while preserving image's visual features. 

## Features
- Detect duplicated images
- Identify visually similar images based on Hamming distance 
    - Customizable similarity threshold for hamming distance computation

- Convert image format 

## Installation
To install project's dependencies, 
```shell
conda env create -f environment.yml

conda activate image_deduplicator  
```

To handle `.HEIC` format images, install the dependency
```shell
python3 -m pip install --upgrade pillow-heif --no-binary :all:

```

## Usage
To remove duplicated images, you can also remove n similar images by adding `--remove_n_similar_img { number of similar image to be removed }`. To define a hamming distance threshold, modify `--hamming_threshold { desired threshold }`, 

```shell
# /src 

python image_deduplicator.py --dir_path {images_dir} --remove_dup True

python image_deduplicator.py --dir_path {images_dir} --remove_dup True --remove_n_similar_img {100} --hamming_threshold {80}
```

To convert image format, run
```
# /src

python utils.py --dir_path {images_dir_path} --output_format {desired extension format e.g .jpg}
```

To crop and resize image, 
```
# /src

python utils.py --dir_path {images_dir_path} --output_dir {output_dir}
```
### Running with Docker
To run the image deduplication package using Docker, `cd` to `image_deduplicate` where the Dockerfile is and follow these steps:

#### Building Docker Image
First, ensure you have Docker installed on your system. Then, build the Docker image using the provided Dockerfile:

```shell
docker build -t image-deduplicator .
```

#### Running the Container
To check for duplicated images, 
```shell
docker run image-deduplication:latest --dir_path test
s/sample_data/
```

#### For debugging
Inspect the container with the following command,
```shell
docker run -it --entrypoint /bin/bash image-deduplica
tion:latest 
```

#### Testing the hash function
To test the hash function in the container
```shell
docker run --rm --
entrypoint pytest image-deduplication -s tests/
```