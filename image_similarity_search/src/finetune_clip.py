import os
import torch
import logging
import clip
import argparse
import yaml
from tqdm import tqdm
from data_prep import image_title_dataset
from torch.utils.data import DataLoader
from typing import Dict, Any
import matplotlib.pyplot as plt


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_up_logger(log_dir=str):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "finetune_clip.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def create_dataloader(img_paths_list):
    img_txt_list = []
    img_path_list = []

    img_text_d = {
        0: "class 1",
        1: "class 2"
    }

    for idx, sub_dir in enumerate(img_paths_list):
        for file in os.listdir(sub_dir):
            img_path_list.append(os.path.join(sub_dir, file))
            img_txt_list.append(img_text_d[idx])
    dataset = image_title_dataset(img_path_list, img_txt_list)
    dataloader = DataLoader(dataset, batch_size=8)
    return dataloader


def create_model_and_optimiser(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.to(device)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=1e-8, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01
    )

    return model, optimiser


def save_checkpoint(
    state: Dict[str, Any], is_best: bool, checkpoint_dir: str, filename: str
):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, f"model_best_{filename}")
        torch.save(state, best_filepath)


def load_checkpoint(checkpoint_dir: str, filename: str, model, optimiser, device):
    filepath = os.path.join(checkpoint_dir, filename)
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimiser.load_state_dict(checkpoint["optimiser"])
        return checkpoint["epoch"], checkpoint["best_loss"]

    return 0, float("inf")


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimiser: torch.optim.Optimizer,
    epoch: int,
    device: str,
    logger: logging.Logger,
):
    total_loss = 0
    loss_img = torch.nn.CrossEntropyLoss()
    loss_txt = torch.nn.CrossEntropyLoss()

    for batch in tqdm(dataloader):
        images, texts = batch
        images, texts = images.to(device), texts.to(device)

        logits_per_image, logits_per_text = model(images, texts)
        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

        loss = (
            loss_img(logits_per_image, ground_truth)
            + loss_txt(logits_per_text, ground_truth)
        ) / 2

        logger.info(f"loss: {loss}")

        if torch.isnan(loss).any():
            logger.error(f"NaN or Inf loss detected at Epoch {epoch}")
            logger.error(f"logits_per_image: {logits_per_image}")
            logger.error(f"logits_per_text: {logits_per_text}")
            raise ValueError("NaN or Inf loss detected. Stopping training.")

        loss.backward()

        if device == "cpu":
            optimiser.step()
        else:
            convert_models_to_fp32(model)
            optimiser.step()
            clip.model.convert_weights(model)

        with torch.no_grad():
            total_loss += loss.item()

        # if i % 10 == 0:  # Log every 10 steps
        #     logger.info(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    logger.info(f"Epoch: {epoch}, Average Loss: {avg_loss:.4f}")
    return avg_loss


class MetricTracker:
    def __init__(self):
        self.metrics = {}

    def update(self, metric_name, value, step):
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append((step, value))

    def get_metric(self, metric_name):
        return self.metrics.get(metric_name, [])


def main(config):
    logger = set_up_logger(config["log_dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric_tracker = MetricTracker()

    # [TODO] setup log dir, set up checkpoint dir
    checkpoint_dir = config["checkpoint_dir"]
    train_dataloader = create_dataloader(config["img_paths_list"])
    model, optimiser = create_model_and_optimiser(config)
    start_epoch, best_loss = load_checkpoint(
        checkpoint_dir, "last_checpoint.pt", model, optimiser, device
    )

    for epoch in range(start_epoch, config["num_epochs"]):
        ep_loss = train_epoch(model, train_dataloader, optimiser, epoch, device, logger)
        metric_tracker.update("loss", ep_loss, epoch)

        # [TODO] save checkpoint
        logger.info(f"Epoch: {epoch}, loss: {ep_loss}")

        is_best = ep_loss < best_loss
        best_loss = min(ep_loss, best_loss)

        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "loss": ep_loss,
            "best_loss": best_loss,
        }

        save_checkpoint(
            checkpoint, is_best, checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"
        )
        save_checkpoint(checkpoint, False, checkpoint_dir, "last_checpoint.pt")

        plt.figure(figsize=(10, 5))
        losses = metric_tracker.get_metric("loss")
        plt.plot([x[0] for x in losses], [x[1] for x in losses])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(config["log_dir"], "loss_curve.png"))
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="clip fine tuning")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
