import logging
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.utils
import torchvision.transforms as transforms
from flwr.common.logger import log
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from transformers import CLIPModel


def resnet50() -> torch.nn.modules:
    resnet50_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    resnet50_model = torch.nn.Sequential(
        *(list(resnet50_model.children())[:-1]), torch.nn.Flatten()
    )

    resnet50_model.hidden_dimension = 2048

    return resnet50_model

def transform(mean: List, std: List) -> transforms.Compose:
    transform_comp = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transform_comp


def extract_features(
    dataloader: DataLoader, feature_extractor: torch.nn.Module, device: torch.device
) -> Tuple[NDArray, NDArray]:

    feature_extractor.to(device)

    features, labels = [], []
    for sample in dataloader:
        batch_samples = sample["img"].to(device)
        batch_label = sample["label"].to(device)
        with torch.no_grad():
            feature = feature_extractor(batch_samples)
        features.append(feature.cpu().detach().numpy())
        labels.append(batch_label.cpu().detach().numpy())

    features_np = np.concatenate(features, axis=0).astype("float64")
    labels_np = np.concatenate(labels)

    return features_np, labels_np

def test(
    classifier_head: torch.nn.Linear,
    dataloader: DataLoader,
    feature_extractor: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:

    classifier_head.eval()
    feature_extractor.eval()
    classifier_head.to(device)
    feature_extractor.to(device)

    correct, total, loss = 0, 0, 0
    for sample in dataloader:
        samples = sample["img"].to(device)
        labels = sample["label"].to(device)
        with torch.no_grad():
            feature = feature_extractor(samples)
            output = classifier_head(feature)
        pred = torch.max(output, 1)[1].data.squeeze()
        correct += (pred == labels).sum().item()
        total += samples.shape[0]
        running_loss = nn.CrossEntropyLoss()(output, labels)
        loss += running_loss.cpu().item()

    return loss, correct / total


def train(
    classifier_head: torch.nn.Linear,
    dataloader: DataLoader,
    opt: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    feature_extractor: Optional[torch.nn.Module] = None,
    verbose: Optional[bool] = False,
) -> None:

    classifier_head.to(device)
    if feature_extractor:
        feature_extractor.eval()
        feature_extractor.to(device)

    for epoch in range(num_epochs):
        correct, total, loss = 0, 0, 0
        for _, batch in enumerate(dataloader):
            classifier_head.zero_grad()
            samples = batch["img"].to(device)
            labels = batch["label"].to(device)
            if feature_extractor:
                with torch.no_grad():
                    samples = feature_extractor(samples)
            output = classifier_head(samples)
            pred = torch.max(output, 1)[1].data.squeeze()
            correct += (pred == labels).sum().item()
            total += samples.shape[0]
            running_loss = nn.CrossEntropyLoss()(output, labels)
            loss += running_loss
            running_loss.backward()
            opt.step()
        if verbose:
            log(logging.INFO, "Epoch: %s --- Accuracy: %s", epoch + 1, correct / total)
