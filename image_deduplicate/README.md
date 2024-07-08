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
```
conda env create -f environment.yml
conda activate image_deduplicator  
```

## Usage
To remove duplicated images, run
```
# /src 

python image_deduplicator.py --dir_path {images_dir} --remove_dup True
```

To convert image format, run
```
# /src

python utils.py --dir_path {images_dir} --output_format {desired extension format e.g .jpg}
```