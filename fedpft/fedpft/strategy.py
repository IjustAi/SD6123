from typing import Dict, List, Optional, Tuple, Union
import torch
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from omegaconf import DictConfig
from sklearn.mixture import GaussianMixture as GMM
from torch.utils.data import DataLoader

from fedpft.models import train
from fedpft.utils import chunks, ndarrays_to_gmmparam

class FedPFT(FedAvg):
    def __init__(
        self,
        *args,
        num_classes: int,
        feature_dimension: int,
        server_opt: DictConfig,
        server_batch_size: int,
        num_epochs: int,
        device: torch.device,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.feature_dimension = feature_dimension
        self.server_opt = server_opt
        self.server_batch_size = server_batch_size
        self.num_epochs = num_epochs
        self.device = device

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not self.accept_failures and failures:
            raise Exception("there are failures and failures are not accepted")

        assert self.on_fit_config_fn is not None
        config = self.on_fit_config_fn(server_round)

        synthetic_features_dataset: List[Union[Dict, Tuple]] = []
        for _, fit_res in results:
            ndarray = parameters_to_ndarrays(fit_res.parameters)
            all_gmm_parameters = [
                ndarrays_to_gmmparam(array) for array in chunks(ndarray, 5)
            ]

            for gmm_parameter in all_gmm_parameters:
                gmm = GMM(
                    n_components=int(config["n_mixtures"]),
                    covariance_type=config["cov_type"],
                    random_state=int(config["seed"]),
                    tol=float(config["tol"]),
                    max_iter=int(config["max_iter"]),
                )
                gmm.means_ = gmm_parameter.means.astype("float32")
                gmm.weights_ = gmm_parameter.weights.astype("float32")
                gmm.covariances_ = gmm_parameter.covariances.astype("float32")

                syn_features, _ = gmm.sample(gmm_parameter.num_samples)
                syn_features = torch.tensor(syn_features, dtype=torch.float32)
                gmm_labels = torch.tensor(
                    [int(gmm_parameter.label)] * int(gmm_parameter.num_samples)
                )

                synthetic_features_dataset += list(zip(syn_features, gmm_labels))

        synthetic_features_dataset = [
            {"img": img, "label": label} for img, label in synthetic_features_dataset
        ]
        synthetic_loader = DataLoader(
            synthetic_features_dataset,
            batch_size=self.server_batch_size,
            shuffle=True,
        )
        classifier_head = torch.nn.Linear(self.feature_dimension, self.num_classes)
        opt = torch.optim.AdamW(
            params=classifier_head.parameters(), lr=self.server_opt.lr
        )

        train(
            classifier_head=classifier_head,
            dataloader=synthetic_loader,
            device=self.device,
            num_epochs=self.num_epochs,
            opt=opt,
            verbose=True,
        )

        classifier_ndarray = [
            val.cpu().numpy() for _, val in classifier_head.state_dict().items()
        ]

        return ndarrays_to_parameters(classifier_ndarray), {}
