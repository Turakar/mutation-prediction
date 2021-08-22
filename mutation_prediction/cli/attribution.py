import argparse
import copy
import functools
import json
import os
import tempfile

import numpy as np
import optuna
import torch
from numpy.lib.format import open_memmap
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

import mutation_prediction.attribution as attribution
from mutation_prediction import cli, utils
from mutation_prediction.cli import registry
from mutation_prediction.data import preprocessing
from mutation_prediction.embeddings.msa import PrecomputedAutoEncoder
from mutation_prediction.embeddings.other import Linearize
from mutation_prediction.embeddings.spectral import OptionalSpectral
from mutation_prediction.models.baseline import GlmnetRegressor
from mutation_prediction.models.classic import EpsilonSvr
from mutation_prediction.models.cnn import KCnn
from mutation_prediction.models.mlp import MlpRegressor


def main():
    parser = argparse.ArgumentParser(description="Tools for attribution.")
    subparsers = parser.add_subparsers(title="subcommands", required=True, dest="subcommand")
    parser.add_argument("--database", "-d", help="Database URL")
    parser.add_argument(
        "--reduce-dataset",
        action="store_true",
        help="Reduces test dataset. Currently implemented for Dr500 only.",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Run attribution on GPU. Does not apply to training."
    )
    parser.add_argument("--load-from", help="Load model from path instead of re-training.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")

    def add_integrated_gradients_params(parser):
        parser.add_argument(
            "--steps",
            type=int,
            default=50,
            help="Number of steps to use for integral approximation.",
        )
        parser.add_argument(
            "--max-error", type=float, default=1e-3, help="Maximum tolerated convergence error."
        )
        parser.add_argument(
            "--batch-int", type=int, default=32, help="Internal batch size for Captum."
        )

    parser_inputs = subparsers.add_parser(
        "inputs", help="Attribute output of a model to its inputs using Integrated Gradients."
    )
    parser_inputs.set_defaults(func=main_inputs)
    add_integrated_gradients_params(parser_inputs)
    parser_inputs.add_argument("study", help="Name of the study.")
    parser_inputs.add_argument(
        "name",
        help="Filename to save attributions to. Saved in 'attributions/<name>.npy' and 'attributions/<name>.json'.",
    )
    parser_inputs.add_argument(
        "--spectral",
        action="store_true",
        help="Do not propagate through the FFT, but stop at spectral values.",
    )

    parser_filter_activations = subparsers.add_parser(
        "filter-activations", help="Record which filters activated where."
    )
    parser_filter_activations.set_defaults(func=main_filter_activations)
    parser_filter_activations.add_argument("study", help="Name of the study.")
    parser_filter_activations.add_argument(
        "name",
        help="Filename to save attributions to. Saved in 'attributions/<name>.npy' and 'attributions/<name>.json'.",
    )

    parser_knn = subparsers.add_parser(
        "knn", help="Attribute kNN inputs of a kCNN using Integrated Gradients."
    )
    parser_knn.set_defaults(func=main_knn)
    add_integrated_gradients_params(parser_knn)
    parser_knn.add_argument("study", help="Name of the study.")
    parser_knn.add_argument(
        "name",
        help="Filename to save attributions to. Saved in 'attributions/<name>.npy' and 'attributions/<name>.json'.",
    )

    parser_filter_attributions = subparsers.add_parser("filter-attribution")
    parser_filter_attributions.set_defaults(func=main_filter_attributions)
    add_integrated_gradients_params(parser_filter_attributions)
    parser_filter_attributions.add_argument("study", help="Name of the study.")
    parser_filter_attributions.add_argument(
        "name",
        help="Filename to save attributions to. Saved in 'attributions/<name>.npy' and 'attributions/<name>.json'.",
    )

    parser_mutations = subparsers.add_parser("mutations")
    parser_mutations.set_defaults(func=main_mutations)
    add_integrated_gradients_params(parser_mutations)
    parser_mutations.add_argument("study", help="Name of the study.")
    parser_mutations.add_argument(
        "name",
        help="Filename to save attributions to. Saved in 'attributions/<name>.npy' and 'attributions/<name>.json'.",
    )
    parser_mutations.add_argument(
        "--half-prottrans", help="Run the ProtTrans model with half precision.", action="store_true"
    )

    args = parser.parse_args()
    args.func(args)


def main_inputs(args):
    complete_dataset, model, study, test, _ = load(args)

    print("Attributing...")

    spectral_flag = args.spectral
    if "MutInd" in study.user_attrs["model"]:
        spectral_flag = True

    if isinstance(model, EpsilonSvr):
        torch_model = attribution.TorchEpsilonSvr(model.model)
    elif isinstance(model, GlmnetRegressor):
        torch_model = attribution.TorchGlmnet(model.model)
    elif isinstance(model, KCnn):
        spectral_flag = True
        torch_model = attribution.TorchKCnn(model)
    elif isinstance(model, MlpRegressor):
        torch_model = attribution.TorchMlp(model)
    else:
        raise ValueError("Unknown model type!")

    device = get_device(args)

    torch_spectral = attribution.TorchPipeline(model.embedding, torch_model).to(device)
    baseline, _ = torch_spectral.embed(
        attribution.extract_wild_type(complete_dataset), spectral=spectral_flag
    )
    x, indices = torch_spectral.embed(test, spectral=spectral_flag)
    attribution.attribute(
        torch_spectral,
        x,
        baseline,
        ext_batch_size=args.batch,
        int_batch_size=args.batch_int,
        n_steps=args.steps,
        mmap_file=os.path.join("attributions", "%s.npy" % args.name),
        max_error=args.max_error,
        forward_args=indices,
        device=device,
    )
    save_metadata(
        args.name,
        dict(
            study=study.study_name,
            hyperparams=model.hyperparams.get(),
        ),
    )
    print("Done.")


def main_filter_activations(args):
    print("Loading...")
    complete_dataset, model, study, test, _ = load(args)
    assert isinstance(model, KCnn)
    device = get_device(args)

    print("Moving model to device...")
    torch_model = attribution.TorchKCnn(model).to(device)

    print("Embedding...")
    x = torch.as_tensor(model.embedding.embed(test), device=device, dtype=torch.float)

    print("Attributing...")
    activated = open_memmap(
        os.path.join("attributions", "%s.npy" % args.name),
        mode="w+",
        dtype=np.uint32,
        shape=(x.shape[0], model.hyperparams["filters"].get()),
    )
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(x), args.batch)):
            batch_end = min(len(x), batch_start + args.batch)
            activation = torch_model(x[batch_start:batch_end], {"input", "filter"})
            where = torch.max(activation, dim=1)[1]
            activated[batch_start:batch_end] = where.cpu().numpy()
    save_metadata(
        args.name,
        dict(
            study=study.study_name,
            hyperparams=model.hyperparams.get(),
        ),
    )


def main_knn(args):
    complete_dataset, model, study, test, _ = load(args)

    print("Attributing...")

    assert isinstance(model, KCnn)

    device = get_device(args)

    torch_model = attribution.TorchKCnn(model).to(device)
    baseline = torch.as_tensor(
        model.embedding.embed(attribution.extract_wild_type(complete_dataset)),
        dtype=torch.float,
        device=device,
    )
    x = torch.as_tensor(model.embedding.embed(test), dtype=torch.float, device=device)
    attribution.attribute(
        torch_model,
        x,
        baseline,
        ext_batch_size=args.batch,
        int_batch_size=args.batch_int,
        n_steps=args.steps,
        mmap_file=os.path.join("attributions", "%s.npy" % args.name),
        max_error=args.max_error,
        forward_args={"filter", "max", "final"},
        device=device,
        preprocessing=functools.partial(torch_model, phases={"input"}),
    )
    save_metadata(
        args.name,
        dict(
            study=study.study_name,
            hyperparams=model.hyperparams.get(),
        ),
    )
    print("Done.")


def main_filter_attributions(args):
    complete_dataset, model, study, test, _ = load(args)

    print("Attributing...")

    assert isinstance(model, KCnn)

    device = get_device(args)

    torch_model = attribution.TorchKCnn(model).to(device)
    baseline = torch.as_tensor(
        model.embedding.embed(attribution.extract_wild_type(complete_dataset)),
        dtype=torch.float,
        device=device,
    )
    x = torch.as_tensor(model.embedding.embed(test), dtype=torch.float, device=device)
    attribution.attribute(
        torch_model,
        x,
        baseline,
        ext_batch_size=args.batch,
        int_batch_size=args.batch_int,
        n_steps=args.steps,
        mmap_file=os.path.join("attributions", "%s.npy" % args.name),
        max_error=args.max_error,
        forward_args={"final"},
        device=device,
        preprocessing=functools.partial(torch_model, phases={"input", "filter", "max"}),
    )
    save_metadata(
        args.name,
        dict(
            study=study.study_name,
            hyperparams=model.hyperparams.get(),
        ),
    )
    print("Done.")


def main_mutations(args):
    print("Loading...")
    complete_dataset, model, study, test, storage = load(args)

    if isinstance(model, EpsilonSvr):
        torch_model = attribution.TorchEpsilonSvr(model.model)
    elif isinstance(model, GlmnetRegressor):
        torch_model = attribution.TorchGlmnet(model.model)
    elif isinstance(model, KCnn):
        torch_model = attribution.TorchKCnn(model)
    elif isinstance(model, MlpRegressor):
        torch_model = attribution.TorchMlp(model)
    else:
        raise ValueError("Unknown model type!")

    device = get_device(args)

    print("Embedding...")
    torch_spectral = attribution.TorchPipeline(
        model.embedding, torch_model, half_prottrans=args.half_prottrans
    ).to(device)
    baseline, _ = torch_spectral.embed(
        attribution.extract_wild_type(complete_dataset),
        spectral=True,
        autoencoder=False,
        prottrans=False,
        storage=storage,
    )
    x, indices = torch_spectral.embed(
        test, spectral=True, autoencoder=False, prottrans=False, storage=storage
    )

    print("Attributing...")
    if torch_spectral.prottrans and len(indices) == 1:
        layers = [torch_spectral.prottrans.model.get_input_embeddings()]
    elif torch_spectral.prottrans:
        layers = [
            torch_spectral.non_prottrans,
            torch_spectral.prottrans.model.get_input_embeddings(),
        ]
    else:
        layers = [torch_spectral.non_prottrans]
    layer_attributions = attribution.attribute(
        torch_spectral,
        x,
        baseline,
        ext_batch_size=args.batch,
        int_batch_size=args.batch_int,
        n_steps=args.steps,
        max_error=args.max_error,
        forward_args=indices,
        device=device,
        layers=layers,
    )
    mutation_attributions = np.zeros_like(test.get_positions(), dtype=np.float32)
    sequence_length = len(test.get_sequence())
    if torch_spectral.prottrans and len(indices) == 1:
        prottrans = torch_spectral.prottrans.reduce_to_sequence(layer_attributions, sequence_length)
        other = np.zeros_like(prottrans)
    elif torch_spectral.prottrans:
        other = layer_attributions[0]
        prottrans = torch_spectral.prottrans.reduce_to_sequence(
            layer_attributions[1], sequence_length
        )
    else:
        other = layer_attributions
        prottrans = np.zeros_like(other)
    for i, mutant in enumerate(preprocessing.dataset_to_tuples(test)):
        mask = utils.make_mask(sequence_length, *list({p for p, a in mutant}))
        mutation_attributions[i, : len(mutant)] = np.sum(other[i][mask], axis=-1) + np.sum(
            prottrans[i][mask], axis=-1
        )
        assert np.allclose(other[i][~mask], 0)
        assert np.allclose(prottrans[i][~mask], 0)
    np.save(os.path.join("attributions", "%s.npy" % args.name), mutation_attributions)

    save_metadata(
        args.name,
        dict(
            study=study.study_name,
            hyperparams=model.hyperparams.get(),
        ),
    )
    print("Done.")


def save_metadata(name, metadata):
    with open(os.path.join("attributions", "%s.json" % name), "w") as fd:
        json.dump(metadata, fd)


def get_device(args):
    if args.gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def load(args):
    storage = cli.url_to_storage(args.database)
    study = optuna.load_study(args.study, storage)
    model = registry.models[study.user_attrs["model"]]()
    dataset_name = study.user_attrs["dataset"]
    train, test = registry.datasets[dataset_name]()
    complete_dataset = train + test
    if args.reduce_dataset:
        train, test = registry.datasets[dataset_name + "_red"]()
    if args.load_from:
        model.load(args.load_from)
        model.embedding.embed_update(train)
    else:
        trial_number = study.user_attrs.get("selected", study.best_trial.number)
        trial = study.trials[trial_number]
        model.hyperparams.set_from_trial(trial, study.user_attrs["params"])
        print("Training...")
        model.fit(train)
        print("Evaluating...")
        pred = model.predict(test)
        print("R² = %.4f" % r2_score(test.get_y(), pred))
    return complete_dataset, model, study, test, storage


if __name__ == "__main__":
    main()
