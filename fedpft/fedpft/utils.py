from dataclasses import dataclass
from typing import List
import numpy as np
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture

@dataclass
class GMMParameters:

    label: NDArray
    means: NDArray
    weights: NDArray
    covariances: NDArray
    num_samples: NDArray


def gmmparam_to_ndarrays(gmm: GMMParameters) -> List[NDArray]:
    return [gmm.label, gmm.means, gmm.weights, gmm.covariances, gmm.num_samples]


def ndarrays_to_gmmparam(ndarrays: NDArray) -> GMMParameters:
    return GMMParameters(
        label=ndarrays[0],
        means=ndarrays[1],
        weights=ndarrays[2],
        covariances=ndarrays[3],
        num_samples=ndarrays[4],
    )

def learn_gmm(
    features: NDArray,
    labels: NDArray,
    n_mixtures: int,
    cov_type: str,
    seed: int,
    tol: float = 1e-12,
    max_iter: int = 1000,
) -> List[GMMParameters]:
   
    gmm_list = []
    for label in np.unique(labels):
        cond_features = features[label == labels]
        if (
            len(cond_features) > n_mixtures
        ):  
            gmm = GaussianMixture(
                n_components=n_mixtures,
                covariance_type=cov_type,
                random_state=seed,
                tol=tol,
                max_iter=max_iter,
            )
            gmm.fit(cond_features)
            gmm_list.append(
                GMMParameters(
                    label=np.array(label),
                    means=gmm.means_.astype("float16"),
                    weights=gmm.weights_.astype("float16"),
                    covariances=gmm.covariances_.astype("float16"),
                    num_samples=np.array(len(cond_features)),
                )
            )
    return gmm_list


def chunks(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]
