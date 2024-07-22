import os
import clip
import torch
import numpy as np
import argparse
import yaml
from open_clip import tokenizer
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple

# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
checkpoint_model = torch.load(
    "/home/mai/Project/breakfast/image_similarity_search/src/checkpoints/model_best_checkpoint_epoch_29.pt"
)


def load_config(config_path: str) -> dict:
    """
    Load config from YAML file.

    Args:
        config_path (str): Path to the YAML file.

    Returns:
        dict: Config dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_top_k(
    model, preprocess, sample_img: str, text_labels: List[str], k: int = 11
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get top k predictions for an image.

    Args:
        model: The model to use for prediction.
        preprocess: The preprocessing function for the image.
        sample_img (str): Path to the sample image.
        text_labels (List[str]): List of text labels.
        k (int): Number of top predictions to return.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Top probabilities and categories.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_tokens = tokenizer.tokenize(text_labels).to(device)
    image_input = preprocess(Image.open(sample_img)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    top_prob, top_cat = text_probs.cpu().topk(k, dim=-1)
    return top_prob, top_cat


def visualize_top_k(
    model, preprocess, sample_img: str, text_labels: List[str], k: int = 11
) -> None:
    """
    Visualize top k predictions for an image.

    Args:
        model: The model to use for prediction.
        preprocess: The preprocessing function for the image.
        sample_img (str): Path to the sample image.
        text_labels (List[str]): List of text labels.
        k (int): Number of top predictions to visualize.
    """
    plt.figure(figsize=(20, 10))

    top_prob, top_cat = get_top_k(model, preprocess, sample_img, text_labels, k=k)

    for i in range(k):
        print(f"{text_labels[top_cat[0][i]]}: {top_prob[0][i]:.2f}")

    _plot_image(sample_img)
    _plot_probabilities(top_prob, top_cat, text_labels)

    plt.tight_layout()
    plt.show()


def _plot_image(sample_img: str) -> None:
    """Plot the sample image."""
    plt.subplot(1, 2, 1)
    plt.imshow(plt.imread(sample_img))
    plt.axis("off")


def _plot_probabilities(
    top_prob: torch.Tensor, top_cat: torch.Tensor, text_labels: List[str]
) -> None:
    """Plot the probabilities as a horizontal bar chart."""
    plt.subplot(1, 2, 2)
    y = np.arange(top_prob.shape[-1])
    plt.grid()
    plt.barh(y, top_prob[0])

    for idx, value in enumerate(top_prob[0]):
        plt.text(value + 0.01, idx, f"{value:.2f}")

    plt.gca().invert_yaxis()
    plt.gca().set_axisbelow(True)
    plt.yticks(y, (text_labels[i] for i in top_cat[0]))
    plt.xlabel("Probability")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="clip inference")
    parser.add_argument(
        "--img_dir",
        type=str,
    )
    args = parser.parse_args()

    infer_img_dir = args.img_dir

    # load config
    config_path = "/home/mai/Project/breakfast/image_similarity_search/config.yaml"
    config = load_config(config_path)

    for img in os.listdir(infer_img_dir):
        sample_img = os.path.join(infer_img_dir, img)
        visualize_top_k(model, preprocess, sample_img, config["text_labels"], k=11)
