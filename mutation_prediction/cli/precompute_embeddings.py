import argparse
import functools
import gc
import os.path
import pickle
from os.path import join as path_join

import numpy as np
import optuna
import torch
from numpy.lib.format import open_memmap
from tqdm import tqdm
from transformers import (
    AlbertTokenizer,
    AutoModel,
    AutoTokenizer,
    T5EncoderModel,
    T5Tokenizer,
    XLNetTokenizer,
    pipeline,
)

import mutation_prediction.data.preprocessing as preprocessing
import mutation_prediction.embeddings.prottrans as prottrans
import mutation_prediction.node as node
from mutation_prediction import data
from mutation_prediction.cli import registry
from mutation_prediction.data import loader

datasets = {
    "A": lambda: loader.load_a(),
    "B": lambda: loader.load_b(),
    "C": lambda: loader.load_c(),
    "D": lambda: loader.load_d(),
    "E": lambda: functools.reduce(lambda a, b: a + b, loader.load_e()),
}


def make_index(args):
    dataset = datasets[args.dataset]()
    index = {}
    for i, mutant in enumerate(preprocessing.dataset_to_tuples(dataset)):
        index[frozenset(mutant)] = i
    ensure_output_dir()
    with open(path_join(node.get_precomputed_path(), args.dataset + ".pkl"), "wb") as fd:
        pickle.dump(index, fd)


def precompute_autoencoder(args):
    storage = optuna.storages.RDBStorage(args.database)
    study = optuna.load_study(args.study, storage)
    study_params = study.user_attrs["params"]
    train, test = registry.datasets[study.user_attrs["dataset"]]()
    model = registry.models[study.user_attrs["model"]]()
    model.hyperparams.set_from_trial(study.best_trial, study_params)
    model.fit(train)
    torch.save(model.model.state_dict(), path_join("models", args.output))
    if args.only_model:
        return
    dataset = datasets[args.dataset]()
    probabilities = model.predict_sequences(dataset.get_sequences())
    ensure_output_dir()
    np.save(path_join(node.get_precomputed_path(), args.output + ".npy"), probabilities)


def precompute_protbert(args):
    dataset = datasets[args.dataset]()
    embedding = prottrans.ProtBertFeatureExtraction()
    embedding.hyperparams.set({"selection": "all"})
    embedded = embedding.embed_update(dataset)
    ensure_output_dir()
    np.save(path_join(node.get_precomputed_path(), "ProtBert-%s.npy" % args.dataset), embedded)


def precompute_prottrans(args):
    # setup feature extraction pipeline
    batch_size = args.batch_size
    tokenizer_generator = AutoTokenizer
    model_generator = AutoModel
    model_name = "Rostlab/" + args.model
    if "t5" in model_name:
        tokenizer_generator = T5Tokenizer
        model_generator = T5EncoderModel
    elif "xlnet" in model_name:
        tokenizer_generator = XLNetTokenizer
    elif "albert" in model_name:
        tokenizer_generator = AlbertTokenizer
    tokenizer = tokenizer_generator.from_pretrained(model_name, do_lower_case=False)
    model = model_generator.from_pretrained(model_name)
    gc.collect()

    if args.download:
        print("Early finishing after download.")
        return

    if args.half:
        model = model.half()

    feature_extractor = pipeline("feature-extraction", tokenizer=tokenizer, model=model, device=0)

    # embed datasets
    parts = args.dataset.split(" ")
    ensure_output_dir()
    for dataset_name in parts:
        if dataset_name in datasets:
            dataset = datasets[dataset_name]()
        else:
            dataset = registry.datasets[dataset_name]()[0]
            np.save(
                path_join(
                    node.get_precomputed_path(), "%s-%s-ids.npy" % (args.model, dataset_name)
                ),
                dataset.get_ids(),
            )
        seq_len = len(dataset.get_sequence())
        sequences = [" ".join(data.sequence_to_string(s)) for s in dataset.get_sequences()]
        embeddings = None
        for batch_start in tqdm(range(0, len(sequences), batch_size)):
            batch_end = min(len(sequences), batch_start + batch_size)
            batch = sequences[batch_start:batch_end]
            for i, embedding in enumerate(feature_extractor(batch)):
                embedding = np.asarray(embedding)
                if "electra" in model_name or "bert" in model_name or "electra" in model_name:
                    embedding = embedding[1 : 1 + seq_len]
                elif "xlnet" in model_name:
                    padded_seq_len = len(embedding)
                    embedding = embedding[padded_seq_len - seq_len - 2 : padded_seq_len - 2]
                elif "t5" in model_name:
                    embedding = embedding[:seq_len]
                else:
                    raise ValueError("Unknown model name! Cannot determine correct indexing.")
                if embeddings is None:
                    embeddings = open_memmap(
                        path_join(
                            node.get_precomputed_path(), "%s-%s.npy" % (args.model, dataset_name)
                        ),
                        mode="w+",
                        dtype=np.float32,
                        shape=(len(sequences),) + embedding.shape,
                    )
                embeddings[batch_start + i] = embedding


def ensure_output_dir():
    path = node.get_precomputed_path()
    if not os.path.isdir(path):
        os.mkdir(path)


def main():
    parser = argparse.ArgumentParser(description="Precompute embeddings.")
    subparsers = parser.add_subparsers(title="subcommands", required=True, dest="subcommand")

    parser_index = subparsers.add_parser("index", help="Create dataset index.")
    parser_index.set_defaults(func=make_index)
    parser_index.add_argument("dataset", type=str, choices=datasets.keys())

    parser_autoencoder = subparsers.add_parser(
        "autoencoder", help="Precompute AutoEncoder probabilities."
    )
    parser_autoencoder.set_defaults(func=precompute_autoencoder)
    parser_autoencoder.add_argument("database", type=str, help="URL to the database.")
    parser_autoencoder.add_argument("study", type=str, help="Name of the optuna study.")
    parser_autoencoder.add_argument("output", type=str, help="Name of the output file.")
    parser_autoencoder.add_argument("dataset", type=str, choices=datasets.keys())
    parser_autoencoder.add_argument("--only-model", action="store_true")

    parser_protbert = subparsers.add_parser("protbert", help="Precompute ProtBert embedding.")
    parser_protbert.set_defaults(func=precompute_protbert)
    parser_protbert.add_argument("dataset", type=str, choices=datasets.keys())

    parser_prottrans = subparsers.add_parser("prottrans", help="Precompute a ProtTrans embedding.")
    parser_prottrans.set_defaults(func=precompute_prottrans)
    parser_prottrans.add_argument("model", type=str)
    parser_prottrans.add_argument("dataset", type=str)
    parser_prottrans.add_argument("--half", action="store_true")
    parser_prottrans.add_argument("--batch-size", type=int, default=8)
    parser_prottrans.add_argument("--download", action="store_true")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
