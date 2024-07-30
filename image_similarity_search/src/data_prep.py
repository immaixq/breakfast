import torch
import clip
import os
from PIL import Image
from torch.utils.data import DataLoader
from inference import load_config

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

class image_title_dataset(torch.utils.data.Dataset):
    def __init__(self, img_paths_list, im_text_list):
        self.img_paths_list = img_paths_list
        self.title = clip.tokenize(im_text_list)

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.img_paths_list[idx]))
        return image, self.title[idx]

    
if __name__ == "__main__":

    # load config file
    config = load_config(config_path="./config.yaml")
    
    IM_TEXT_LIST = []
    IM_TEXT_LIST_D = {
        0: "ultramilk_fullcream",
        1: "ultramilk_karamel",
        2: "ultramilk_taro",
        3: "ultramilk_lowfat",
        4: "ultramilk_lowfat_cokelat",
        5: "ultramilk_mocha",
        6: "ultramilk_strawberry",
        7: "ultramilk_mini_kids_cokelat",
        8: "ultramilk_mini_kids_fullcream",
        9: "ultramilk_mini_kids_stroberi",
        10: "ultramilk_mini_kids_vanilla",
    }

    IMGS_PATHS_LIST = []

    for idx, im_path in enumerate(config["img_paths_list"]):
        for im_name in os.listdir(im_path):
            IMGS_PATHS_LIST.append(os.path.join(im_path, im_name))
            IM_TEXT_LIST.append(IM_TEXT_LIST_D[idx])

    dataset = image_title_dataset(IMGS_PATHS_LIST, IM_TEXT_LIST)
    img, title = dataset[0]
    print(img.shape, title)