from typing import Callable, Dict
from flwr_datasets.federated_dataset import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision import transforms

class Dataset:
    def __init__(
        self,
        dataset: str,
        num_clients: int,
        batch_size: int,
        dirichlet_alpha: float,
        partition_by: str,
        image_column_name: str,
        transform: transforms,
        image_input_size: int,
        seed: int = 0,
        split_size: float = 0.8,
        **kwargs,
    ) -> None:
        
        self.dataset = dataset
        self.num_clients = num_clients
        self.image_input_size = image_input_size
        self.transform = transform
        self.batch_size = batch_size
        self.dirichlet_alpha = dirichlet_alpha
        self.partition_by = partition_by
        self.seed = seed
        self.split_size = split_size
        self.image_column_name = image_column_name
        self.kwargs = kwargs

    def get_loaders(self):
        partitioner = DirichletPartitioner(
            num_partitions=self.num_clients,
            partition_by=self.partition_by,
            alpha=self.dirichlet_alpha,
            min_partition_size=10,
            self_balancing=True,
        )

        fds = FederatedDataset(
            dataset=self.dataset, partitioners={"train": partitioner}
        )
        trainloaders, testloaders = [], []
        for partition_id in range(self.num_clients):
            partition = fds.load_partition(partition_id)
            partition = partition.with_transform(self.apply_batch_transforms())
            partition = partition.train_test_split(
                train_size=self.split_size, seed=self.seed
            )
            trainloaders.append(
                DataLoader(partition["train"], batch_size=self.batch_size)
            )
            testloaders.append(
                DataLoader(partition["test"], batch_size=self.batch_size)
            )

        return trainloaders, testloaders

    def apply_batch_transforms(self) -> Callable[[Dict], Dict]:
        
        def batch_transform(batch):
            batch_img = [
                self.transform(
                    img.resize((self.image_input_size, self.image_input_size))
                )
                for img in batch[self.image_column_name]
            ]
            batch_label = list(batch[self.partition_by])

            return {"img": batch_img, "label": batch_label}

        return batch_transform
