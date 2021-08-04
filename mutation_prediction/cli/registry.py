import collections
import functools

import numpy as np

import mutation_prediction.data.loader as loader
import mutation_prediction.data.preprocessing as preprocessing
import mutation_prediction.models.baseline as baseline
import mutation_prediction.models.classic as classic
import mutation_prediction.models.cnn as cnn
import mutation_prediction.models.mlp as mlp
import mutation_prediction.models.msa as msa
import mutation_prediction.models.self_optimized as self_optimized
import mutation_prediction.models.stub as stub
import mutation_prediction.utils as utils
from mutation_prediction.data import Dataset
from mutation_prediction.embeddings.acids import (
    VHSE,
    AcidsOneHot,
    PcScores,
    ProtVec,
    SScales,
    ZScales,
)
from mutation_prediction.embeddings.msa import (
    MlpVariationalAutoEncoder,
    MsaPca,
    PrecomputedAutoEncoder,
)
from mutation_prediction.embeddings.other import ConcatEmbedding, Linearize, MutInd
from mutation_prediction.embeddings.positional import PositionalEmbedding
from mutation_prediction.embeddings.prottrans import (
    ProtBertFeatureExtraction,
    ProtTransFeatureExtractionPrecomputed,
)
from mutation_prediction.embeddings.spectral import (
    OptionalSpectral,
    Spectral,
    SpectralSingle,
)
from mutation_prediction.embeddings.structural import SPairs
from mutation_prediction.models import gnn

# manually selected models
models = {
    "CNNLinear": cnn.CnnRegressorLinear,
    "InnovSARlike": (lambda: baseline.GlmnetRegressor(SpectralSingle())),
    "MlpVaeSvm": (lambda: baseline.SvmRegressor(MlpVariationalAutoEncoder())),
    "MlpVaePls": (lambda: classic.PlsRegressor(MlpVariationalAutoEncoder())),
    "MlpVaeSelfSvm": (lambda: self_optimized.SelfOptimizedSvm(MlpVariationalAutoEncoder())),
    "MlpVaeStub": stub.MlpVariationalAutoEncoderStub,
    "SvmSpectral": (lambda: baseline.SvmRegressor(Spectral(PcScores()))),
    "SelfSvmMsaPca": (lambda: self_optimized.SelfOptimizedSvm(MsaPca())),
    "SelfSvmProtBert": (lambda: self_optimized.SelfOptimizedSvm(ProtBertFeatureExtraction())),
    "MlpProtBert": (lambda: mlp.MlpRegressor(ProtBertFeatureExtraction())),
    "CnnLinearProtBert": (
        lambda: cnn.CnnRegressorLinearCustomEmbedding(ProtBertFeatureExtraction())
    ),
    "Cnn3d": cnn.CnnRegressor3d,
    "MlpAutoEncoder": msa.MlpAutoEncoder,
    "LinearAutoEncoder": msa.LinearAutoEncoder,
    "SvmProb": (lambda: baseline.SvmRegressor(PrecomputedAutoEncoder("Linear"))),
    "CnnLinearSScalesProb": lambda: cnn.CnnRegressorLinearCustomEmbedding(
        ConcatEmbedding(SScales(), PrecomputedAutoEncoder("Linear"))
    ),
    "GraphTransformer": gnn.GraphTransformerRegressor,
    "GraphTransformerSimple": gnn.GraphTransformerSimpleRegressor,
    "GraphConv": gnn.GraphConvRegressor,
    "KCnn": lambda: cnn.KCnn(SScales()),
    "KCnnPlus": cnn.KCnnPlus,
    "KCnnConf": lambda: cnn.KCnn(
        ConcatEmbedding(
            AcidsOneHot(),
            SScales(),
            PrecomputedAutoEncoder("Linear"),
            PositionalEmbedding(),
            configurable=True,
        )
    ),
    "KCnnMlp": lambda: cnn.KCnnMlp(
        ConcatEmbedding(
            AcidsOneHot(),
            SScales(),
            PrecomputedAutoEncoder("Linear"),
            PositionalEmbedding(),
            configurable=True,
        ),
        Linearize(
            ConcatEmbedding(
                AcidsOneHot(),
                SScales(),
                PrecomputedAutoEncoder("Linear"),
                configurable=True,
            )
        ),
    ),
    "EpsilonSvrIdentity": lambda: classic.EpsilonSvr(AcidsOneHot()),
    "GlmnetMutInd": lambda: baseline.GlmnetRegressor(MutInd()),
    "LinearEpsilonSvrMutInd": lambda: classic.EpsilonSvrLinear(MutInd()),
    "EpsilonSvrMutInd": lambda: classic.EpsilonSvr(MutInd()),
    "MlpMutInd": lambda: mlp.MlpRegressor(MutInd()),
    "GlmnetConf": lambda: baseline.GlmnetRegressor(
        ConcatEmbedding(
            Linearize(OptionalSpectral(AcidsOneHot())),
            Linearize(OptionalSpectral(SScales())),
            Linearize(OptionalSpectral(PrecomputedAutoEncoder("Linear"))),
            Linearize(OptionalSpectral(ProtTransFeatureExtractionPrecomputed())),
            configurable=True,
        )
    ),
    "LinearEpsilonSvrConf": lambda: classic.EpsilonSvrLinear(
        ConcatEmbedding(
            Linearize(OptionalSpectral(AcidsOneHot())),
            Linearize(OptionalSpectral(SScales())),
            Linearize(OptionalSpectral(PrecomputedAutoEncoder("Linear"))),
            Linearize(OptionalSpectral(ProtTransFeatureExtractionPrecomputed())),
            configurable=True,
        )
    ),
    "EpsilonSvrConf": lambda: classic.EpsilonSvr(
        ConcatEmbedding(
            Linearize(OptionalSpectral(AcidsOneHot())),
            Linearize(OptionalSpectral(SScales())),
            Linearize(OptionalSpectral(PrecomputedAutoEncoder("Linear"))),
            Linearize(OptionalSpectral(ProtTransFeatureExtractionPrecomputed())),
            configurable=True,
        )
    ),
    "MlpConf": lambda: mlp.MlpRegressor(
        ConcatEmbedding(
            Linearize(OptionalSpectral(AcidsOneHot())),
            Linearize(OptionalSpectral(SScales())),
            Linearize(OptionalSpectral(PrecomputedAutoEncoder("Linear"))),
            Linearize(OptionalSpectral(ProtTransFeatureExtractionPrecomputed())),
            configurable=True,
        )
    ),
    "kCnnConf": lambda: cnn.KCnn(
        ConcatEmbedding(
            AcidsOneHot(),
            SScales(),
            PrecomputedAutoEncoder("Linear"),
            ProtTransFeatureExtractionPrecomputed(),
            configurable=True,
        )
    ),
}

# embedding comparison models
comparison_embeddings = [
    AcidsOneHot,
    SScales,
    functools.partial(PrecomputedAutoEncoder, name="Linear"),
    ProtTransFeatureExtractionPrecomputed,
]
for index in range(1, 2 ** len(comparison_embeddings)):
    binary_str = format(index, "0%db" % len(comparison_embeddings))
    binary_bool = [digit == "1" for digit in binary_str]
    models["EpsilonSvr_%s" % binary_str] = lambda binary_bool=binary_bool: classic.EpsilonSvr(
        ConcatEmbedding(
            *[
                Linearize(OptionalSpectral(embedding()))
                for embedding, selected in zip(comparison_embeddings, binary_bool)
                if selected
            ]
        )
    )

# baseline comparison models
fixed_embedding_models = {
    "BaseGLMNET": baseline.GlmnetRegressor,
    "BaseSVM": baseline.SvmRegressor,
    "BaseRF": baseline.RandomForestRegressor,
    "BaseXGB": baseline.XGBoostRegressor,
    "BaseMLP": baseline.MlpRegressor,
    "BaseCNN": baseline.CnnRegressor,
    "MLP": mlp.MlpRegressor,
}
embeddings = {
    "Identity": AcidsOneHot,
    "zScales": ZScales,
    "VHSE": VHSE,
    "PCscores": PcScores,
    "sScales": SScales,
    "sPairs": SPairs,
    "protVec": ProtVec,
    "mutInd": MutInd,
}
for m_k, m_v in fixed_embedding_models.items():
    for e_k, e_v in embeddings.items():
        if m_k == "BaseCNN" and e_k not in ["Identity", "PCscores", "VHSE", "zScales"]:
            continue
        models[m_k + "_" + e_k] = lambda m_v=m_v, e_v=e_v: m_v(e_v())


def dataset_e():
    train, val, test = loader.load_e()
    val = preprocessing.shuffle(val, seed=42)
    val, test = preprocessing.split_by_index(val + test, training_fraction=0.5)
    return train, val, test


def dataset_e50():
    train, val, test = loader.load_e()
    train, test = preprocessing.split_by_index(
        preprocessing.shuffle(train + val, seed=42) + test, training_fraction=0.5
    )
    return train, test


def dataset_e75():
    train, val, test = loader.load_e()
    train, test = preprocessing.split_by_index(
        preprocessing.shuffle(train + val, seed=42) + test, training_fraction=0.75
    )
    return train, test


def dataset_dcov(k):
    dataset = loader.load_d()
    tupled = preprocessing.dataset_to_tuples(dataset)
    single_coverage_counter = collections.Counter()
    for mutant in tupled:
        single_coverage_counter.update(mutant)
    mutations = {m for m, _ in single_coverage_counter.most_common(k)}
    mask_train = np.zeros((len(dataset),), dtype=np.bool8)
    mask_val_test = np.zeros((len(dataset),), dtype=np.bool8)
    for i, mutant in enumerate(tupled):
        if mutant.issubset(mutations):
            if len(mutant) <= 1:
                mask_train[i] = True
            else:
                mask_val_test[i] = True
    train = dataset[mask_train]
    val, test = preprocessing.split_by_index(
        preprocessing.shuffle(dataset[mask_val_test], seed=42), training_fraction=0.5
    )
    return train, val, test


datasets = {
    # Xu et al. 2020
    "A": lambda: preprocessing.split_by_index(loader.load_a(), 0.765),
    "B": lambda: preprocessing.split_by_index(loader.load_b(), 0.895),
    "C": lambda: preprocessing.split_num_mutations(loader.load_c(), 5),
    "D": lambda: preprocessing.split_sample_mutations(
        loader.load_d(), utils.make_mask(16, 0, 1, 2), 500
    ),
    # InnovSAR
    "E": dataset_e,
    # custom
    "D50": lambda: preprocessing.split_by_index(preprocessing.shuffle(loader.load_d()), 0.5),
    "Dn1": lambda: preprocessing.split_num_mutations(loader.load_d(), 1),
    "Dcov50": lambda: dataset_dcov(50),
    "A75": lambda: preprocessing.split_by_index(
        preprocessing.shuffle(loader.load_a(), seed=42), 0.75
    ),
    "B50": lambda: preprocessing.split_by_index(
        preprocessing.shuffle(loader.load_b(), seed=42), 0.5
    ),
    "B75": lambda: preprocessing.split_by_index(
        preprocessing.shuffle(loader.load_b(), seed=42), 0.75
    ),
    "C50": lambda: preprocessing.split_by_index(
        preprocessing.shuffle(loader.load_c(), seed=42), 0.5
    ),
    "C75": lambda: preprocessing.split_by_index(
        preprocessing.shuffle(loader.load_c(), seed=42), 0.75
    ),
    "E50": dataset_e50,
    "E75": dataset_e75,
}

# DrN datasets (random subsamples of dataset D of size N)
for i in (50, 100, 200, 500, 1000, 5000, 10000):
    datasets["Dr%d" % i] = lambda i=i: preprocessing.split_by_index_num(
        preprocessing.shuffle(loader.load_d(), seed=42), i
    )


def split_by_slice(dataset, start, end):
    mask = utils.make_mask(len(dataset))
    mask[start:end] = True
    return dataset[mask], dataset[~mask]


# DrN_i datasets (random subsamples of dataset D of size N)
for size in (50, 100, 200, 500, 1000, 2000, 5000, 10000):
    for i in range(min(100, 51715 // size)):
        datasets["Dr%d_%d" % (size, i)] = lambda size=size, i=i: split_by_slice(
            preprocessing.shuffle(loader.load_d(), seed=42), i * size, (i + 1) * size
        )


# full datasets (complete dataset in both train and test)
def full_dataset(dataset):
    if isinstance(dataset, tuple):
        dataset = functools.reduce(lambda a, b: a + b, dataset)
    return dataset, dataset


for name, loader_fn in {
    "A": loader.load_a,
    "B": loader.load_b,
    "C": loader.load_c,
    "D": loader.load_d,
    "E": loader.load_e,
}.items():
    datasets["%sfull" % name] = lambda loader_fn=loader_fn: full_dataset(loader_fn())


# XsP_i datasets (random vs. #mutations-based sampling)
def xsp_dataset(dataset: Dataset, p: float, seed: int, training_fraction: float = 0.75):
    # randomize
    dataset = preprocessing.shuffle(dataset, seed=seed)
    # determine the numbers
    num_training = int(len(dataset) * training_fraction)
    num_by_mutation = int(num_training * p)
    num_random = num_training - num_by_mutation
    # split by mutation count
    sorted_mutants = np.argsort(dataset.get_num_mutations())
    indices_by_mutation = sorted_mutants[:num_by_mutation]
    # split random
    rng = np.random.default_rng()
    indices_random = rng.choice(sorted_mutants[num_by_mutation:], size=num_random, replace=False)
    # split
    combined_indices = np.concatenate([indices_by_mutation, indices_random])
    mask = np.zeros(len(dataset), dtype=np.bool_)
    mask[combined_indices] = True
    return dataset[mask], dataset[~mask]


for name, loader_fn in {
    "A": datasets["Afull"],
    "B": datasets["Bfull"],
    "C": datasets["Cfull"],
    "E": datasets["Efull"],
}.items():
    for p in [0, 25, 50, 75, 100]:
        for seed in range(20):
            datasets["%ss%d_%d" % (name, p, seed)] = lambda loader_fn=loader_fn, p=p: xsp_dataset(
                loader_fn()[0], p / 100.0, seed
            )
