#!/usr/bin/env python
#  coding=utf-8

"""
Creates a new model from a set of options.
"""

import argparse
from pathlib import Path

import models.model as mm
import models.vocabulary as mv
import models.decorator as md
import utils.chem as uc
import utils.log as ul


def parse_args():
    """Parses arguments from cmd"""
    parser = argparse.ArgumentParser(description="Create a model with the vocabulary extracted from a SMILES file.")

    parser.add_argument("--input-smiles-path", "-i",
                        help=("File with two fields (scaffold, decoration) to calculate the vocabularies from.\
                        The SMILES are taken as-is, no processing is done."),
                        type=str, default= "./data/randomized_smiles/000.smi")
    parser.add_argument("--output-model-path", "-o", help="Prefix or path to the output model file.", type=str, default= "./pretrained_models")
    parser.add_argument("--num-layers", "-l",
                        help="Number of RNN layers of the model [DEFAULT: 3]", type=int, default=3)
    parser.add_argument("--layer-size", "-s",
                        help="Size of each of the RNN layers [DEFAULT: 512]", type=int, default=512)
    parser.add_argument("--embedding-layer-size", "-e",
                        help="Size of the embedding layer [DEFAULT: 256]", type=int, default=256)
    parser.add_argument("--dropout", "-d",
                        help="Amount of dropout between the GRU layers [DEFAULT: 0.0]", type=float, default=0)
    parser.add_argument("--layer-normalization", "--ln",
                        help="Add layer normalization to the GRU output", action="store_true", default=False)
    parser.add_argument("--max-sequence-length",
                        help="Maximum length of the sequences [DEFAULT: 256]", type=int, default=256)
    parser.add_argument("--scaffold-vocab-output",
                        help="File to save scaffold vocabulary tokens. Defaults to '<output-model-path>_scaffold_tokens.txt'",
                        type=str, default=None)
    parser.add_argument("--decoration-vocab-output",
                        help="File to save decoration vocabulary tokens. Defaults to '<output-model-path>_decoration_tokens.txt'",
                        type=str, default=None)

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    scaffold_list, decoration_list = zip(*uc.read_csv_file(args.input_smiles_path, num_fields=2))

    LOG.info("Building vocabulary")

    vocabulary = mv.DecoratorVocabulary.from_lists(scaffold_list, decoration_list)

    scaffold_tokens = vocabulary.scaffold_vocabulary.tokens()
    decoration_tokens = vocabulary.decoration_vocabulary.tokens()

    LOG.info("Scaffold vocabulary contains %d tokens: %s",
             vocabulary.len_scaffold(), scaffold_tokens)
    LOG.info("Decorator vocabulary contains %d tokens: %s",
             vocabulary.len_decoration(), decoration_tokens)

    _save_tokens(
        tokens=scaffold_tokens,
        explicit_path=args.scaffold_vocab_output,
        default_path=f"{args.output_model_path}_scaffold_tokens.txt",
        description="scaffold"
    )
    _save_tokens(
        tokens=decoration_tokens,
        explicit_path=args.decoration_vocab_output,
        default_path=f"{args.output_model_path}_decoration_tokens.txt",
        description="decoration"
    )

    encoder_params = {
        "num_layers": args.num_layers,
        "num_dimensions": args.layer_size,
        "vocabulary_size": vocabulary.len_scaffold(),
        "dropout": args.dropout
    }

    decoder_params = {
        "num_layers": args.num_layers,
        "num_dimensions": args.layer_size,
        "vocabulary_size": vocabulary.len_decoration(),
        "dropout": args.dropout
    }

    decorator = md.Decorator(encoder_params, decoder_params)
    model = mm.DecoratorModel(vocabulary, decorator, args.max_sequence_length)

    model_output_path = _resolve_model_output_path(args.output_model_path)
    LOG.info("Saving model at %s", model_output_path)
    model.save(str(model_output_path))


LOG = ul.get_logger(name="create_model")


def _save_tokens(tokens, explicit_path, default_path, description):
    """Persist a list of tokens to disk."""
    target_path = Path(explicit_path) if explicit_path else Path(default_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as file_obj:
        for token in tokens:
            file_obj.write(f"{token}\n")
    LOG.info("Saved %s tokens to %s", description, target_path)


def _resolve_model_output_path(output_path):
    """Resolve the model output path to a writable file and ensure parent dirs exist."""
    path = Path(output_path)

    # If a directory is provided explicitly (existing or trailing slash), create a default filename inside it.
    if path.is_dir() or str(output_path).endswith(("/", "\\")):
        path.mkdir(parents=True, exist_ok=True)
        return path / "decorator_model.pt"

    # If no suffix is provided, treat path as prefix and add .pt extension.
    if path.suffix == "":
        path = path.with_suffix(".pt")

    # Ensure parent directory exists for the target file path.
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    main()
