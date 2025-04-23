"""Dataset loading and processing utilities."""

import random
import pickle
from typing import List, Tuple

from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from fedpara.dataset_preparation import (
    DatasetSplit,
    iid,
    noniid,
    noniid_partition_loader,
)

from datasets import load_dataset
from torchvision.transforms import ToTensor
from collections import defaultdict
from PIL import Image
import torch


def load_isic2019_dataloaders(batch_size: int, num_clients: int = 4):
    from datasets import load_dataset
    from torchvision import transforms
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    print("🔄 Loading ISIC2019 from Hugging Face...")

    dataset = load_dataset("flwrlabs/fed-isic2019")
    train_data = list(dataset["train"])  # 转成 list 以支持 shuffle
    test_data = dataset["test"]

    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # ✅ 改图像尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])

    # ✅ 随机划分
    random.shuffle(train_data)
    total = len(train_data)
    samples_per_client = total // num_clients
    train_loaders = []

    for i in range(num_clients):
        start = i * samples_per_client
        end = (i + 1) * samples_per_client if i < num_clients - 1 else total
        client_samples = train_data[start:end]

        images, labels = [], []
        for sample in client_samples:
            img = sample["image"]
            img = img.convert("RGB") if img.mode != "RGB" else img
            images.append(transform(img))
            labels.append(torch.tensor(sample["label"]))
        tensor_dataset = TensorDataset(torch.stack(images), torch.stack(labels))
        train_loaders.append(DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True, num_workers=0))
        print(f"✅ Client {i}: {len(images)} samples")

    # 测试集
    test_images, test_labels = [], []
    for sample in test_data:
        img = sample["image"]
        img = img.convert("RGB") if img.mode != "RGB" else img
        test_images.append(transform(img))
        test_labels.append(torch.tensor(sample["label"]))
    test_dataset = TensorDataset(torch.stack(test_images), torch.stack(test_labels))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loaders, test_loader



def load_datasets(
    config, num_clients, batch_size
) -> Tuple[List[DataLoader], DataLoader]:
    """Load the dataset and return the dataloaders for the clients and the server."""
    print("Loading data...")
    match config["name"]:
        case "CIFAR10":
            Dataset = datasets.CIFAR10
        case "CIFAR100":
            Dataset = datasets.CIFAR100
        case "MNIST":
            Dataset = datasets.MNIST
        case "ISIC2019":
            return load_isic2019_dataloaders(batch_size)
        case _:
            raise NotImplementedError
    data_directory = f"./data/{config['name'].lower()}/"
    match config["name"]:
        case "ISIC2019":
            dataset = load_dataset("flwrlabs/fed-isic2019")
            train_loaders = []
            for client_data in dataset["train"]:
                # 每个 client 是一个字典：{ "image": ..., "label": ... }
                images = [ToTensor()(Image.open(img_path).convert("RGB")) for img_path in client_data["image"]]
                labels = [torch.tensor(label) for label in client_data["label"]]
                tensor_dataset = torch.utils.data.TensorDataset(
                    torch.stack(images), torch.stack(labels)
                )
                train_loaders.append(DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True))

            # 构建测试集（使用合并后的）
            test_data = dataset["test"]
            test_images = [ToTensor()(Image.open(img_path).convert("RGB")) for img_path in test_data["image"]]
            test_labels = [torch.tensor(label) for label in test_data["label"]]
            test_dataset = torch.utils.data.TensorDataset(
                torch.stack(test_images), torch.stack(test_labels)
            )
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            return train_loaders, test_loader
        case "CIFAR10" | "CIFAR100":
            ds_path = f"{data_directory}train_{num_clients}_{config.alpha:.2f}.pkl"
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
            try:
                with open(ds_path, "rb") as file:
                    train_datasets = pickle.load(file).values()
                dataset_train = Dataset(
                    data_directory,
                    train=True,
                    download=False,
                    transform=transform_train,
                )
                dataset_test = Dataset(
                    data_directory,
                    train=False,
                    download=False,
                    transform=transform_test,
                )
            except FileNotFoundError:
                dataset_train = Dataset(
                    data_directory, train=True, download=True, transform=transform_train
                )
                if config.partition == "iid":
                    train_datasets = iid(dataset_train, num_clients)
                else:
                    train_datasets, _ = noniid(dataset_train, num_clients, config.alpha)
                pickle.dump(train_datasets, open(ds_path, "wb"))
                train_datasets = train_datasets.values()
                dataset_test = Dataset(
                    data_directory, train=False, download=True, transform=transform_test
                )

        case "MNIST":
            ds_path = f"{data_directory}train_{num_clients}.pkl"
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
            transform_test = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
            try:
                train_datasets = pickle.load(open(ds_path, "rb"))
                dataset_train = Dataset(
                    data_directory,
                    train=True,
                    download=False,
                    transform=transform_train,
                )
                dataset_test = Dataset(
                    data_directory,
                    train=False,
                    download=False,
                    transform=transform_test,
                )

            except FileNotFoundError:
                dataset_train = Dataset(
                    data_directory, train=True, download=True, transform=transform_train
                )
                train_datasets = noniid_partition_loader(
                    dataset_train,
                    m_per_shard=config.shard_size,
                    n_shards_per_client=len(dataset_train) // (config.shard_size * 100),
                )
                pickle.dump(train_datasets, open(ds_path, "wb"))
                dataset_test = Dataset(
                    data_directory, train=False, download=True, transform=transform_test
                )
            train_loaders = [
                DataLoader(x, batch_size=batch_size, shuffle=True)
                for x in train_datasets
            ]
            test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=2)
            return train_loaders, test_loader

    test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=2)
    train_loaders = [
        DataLoader(
            DatasetSplit(dataset_train, ids),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
        )
        for ids in train_datasets
    ]

    return train_loaders, test_loader
