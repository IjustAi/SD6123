from collections import OrderedDict
from typing import Callable, Dict, List, Tuple
import flwr as fl
import torch
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from fedpft.models import extract_features, test, train
from fedpft.utils import gmmparam_to_ndarrays, learn_gmm


class FedPFTClient(fl.client.NumPyClient):
    def __init__(
        self,
        trainloader: DataLoader,
        testloader: DataLoader,
        feature_extractor: torch.nn.Module,
        num_classes: int,
        device: torch.device,
    ) -> None:

        self.trainloader = trainloader
        self.testloader = testloader
        self.feature_extractor = feature_extractor
        self.classifier_head = nn.Linear(
            feature_extractor.hidden_dimension, num_classes
        )
        self.device = device

    def get_parameters(self, config) -> NDArrays:
        return [
            val.cpu().numpy() for _, val in self.classifier_head.state_dict().items()
        ]

    def set_parameters(self, parameters: NDArrays) -> None:
        params_dict = zip(self.classifier_head.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.classifier_head.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:

        features, labels = extract_features(
            dataloader=self.trainloader,
            feature_extractor=self.feature_extractor,
            device=self.device,
        )

        gmm_list = learn_gmm(
            features=features,
            labels=labels,
            n_mixtures=int(config["n_mixtures"]),
            cov_type=str(config["cov_type"]),
            seed=int(config["seed"]),
            tol=float(config["tol"]),
            max_iter=int(config["max_iter"]),
        )

        return [array for gmm in gmm_list for array in gmmparam_to_ndarrays(gmm)], 0, {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
    
        self.set_parameters(parameters)
        loss, acc = test(
            classifier_head=self.classifier_head,
            dataloader=self.testloader,
            feature_extractor=self.feature_extractor,
            device=self.device,
        )
        return loss, len(self.testloader.dataset), {"accuracy": acc}


def generate_client_fn(
    client_cfg: DictConfig,
    trainloaders: List[DataLoader],
    testloaders: List[DataLoader],
    feature_extractor: torch.nn.Module,
    num_classes: int,
    device: torch.device,
) -> Callable[[str], fl.client.NumPyClient]:

    def client_fn(cid: str) -> fl.client.NumPyClient:

        return instantiate(
            client_cfg,
            trainloader=trainloaders[int(cid)],
            testloader=testloaders[int(cid)],
            feature_extractor=feature_extractor,
            num_classes=num_classes,
            device=device,
        )

    return client_fn