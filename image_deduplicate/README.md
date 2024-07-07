## Introduction 
This python package can be used to find duplicated images and similar images. 

The package uses perceptual hashing, a technique that converts images into compact binary representations while preserving image's visual features. 

## Features
- Detect duplicated images
- Identify visually similar images based on Hamming distance 
    - Customizable similarity threshold for hamming distance computation

## Installation
To install project's dependencies, 
```
conda env create -f environment.yml
conda activate image_deduplicator  
```

## Usage
To remove duplicated images, run
```
python image_deduplicator.py --dir_path {images_dir} --remove_dup True
```
