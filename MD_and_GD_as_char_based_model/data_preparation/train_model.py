#!/usr/bin/env python
#  coding=utf-8

"""
Script to train a model
"""

import argparse
import os.path
import glob
import itertools as it

import torch
import torch.utils.tensorboard as tbx

import collect_stats_from_model as csfm

import models.model as mm
import models.actions as ma

import utils.chem as uc
import utils.log as ul


# Default parameters for running this script without passing CLI arguments.
# Update the paths below to match your local workspace when needed.
DEFAULT_OTHER_PARAMS = {
    "input_model_path": "./pretrained_models/decorator_model.pt",
    "output_model_prefix_path": "./pretrained_models/trained_decorator_model",
    "training_set_path": "./data/randomized_smiles",
    "save_every_n_epochs": 1,
    "epochs": 10,
    "batch_size": 128,
    "clip_gradients": 1.0,
    "collect_stats_frequency": 1,
    "device": "cuda",
}

DEFAULT_COLLECT_STATS_PARAMS = {
    "log_path": "./logs/train_model",
    "validation_set_path": "./data/valid.smi",
    "decoration_type": "single",
    "sample_size": 5,
    "with_weights": False,
    "frequency": 1,
}

DEFAULT_LEARNING_RATE_PARAMS = {
    "start": 1E-4,
    "min": 1E-6,
    "gamma": 0.95,
    "step": 1,
}


class TrainModelPostEpochHook(ma.TrainModelPostEpochHook):

    WRITER_CACHE_EPOCHS = 25

    def __init__(self, output_prefix_path, epochs, validation_sets, lr_scheduler, collect_stats_params,
                 lr_params, collect_stats_frequency, save_frequency, logger=None):
        ma.TrainModelPostEpochHook.__init__(self, logger)

        self.validation_sets = validation_sets
        self.lr_scheduler = lr_scheduler

        self.output_prefix_path = output_prefix_path
        self.save_frequency = save_frequency
        self.epochs = epochs
        self.log_path = collect_stats_params["log_path"]

        self.collect_stats_params = collect_stats_params
        self.collect_stats_frequency = collect_stats_frequency

        self.lr_params = lr_params

        self._writer = None
        if self.collect_stats_frequency > 0:
            self._reset_writer()

    def __del__(self):
        self._close_writer()

    def run(self, model, training_set, epoch):
        if self.collect_stats_frequency > 0 and epoch % self.collect_stats_frequency == 0:
            validation_set = next(self.validation_sets)
            other_values = {"lr": self.get_lr()}

            ma.CollectStatsFromModel(
                model=model, epoch=epoch, training_set=training_set,
                validation_set=validation_set, writer=self._writer, other_values=other_values, logger=self.logger,
                sample_size=self.collect_stats_params["sample_size"]
            ).run()

        self.lr_scheduler.step(epoch=epoch)

        lr_reached_min = (self.get_lr() < self.lr_params["min"])
        if lr_reached_min or self.epochs == epoch \
                or (self.save_frequency > 0 and (epoch % self.save_frequency == 0)):
            model.save(self._model_path(epoch))

        if self._writer and (epoch % self.WRITER_CACHE_EPOCHS == 0):
            self._reset_writer()

        return not lr_reached_min

    def get_lr(self):
        return self.lr_scheduler.optimizer.param_groups[0]["lr"]

    def _model_path(self, epoch):
        return "{}.{}".format(self.output_prefix_path, epoch)

    def _reset_writer(self):
        self._close_writer()
        self._writer = tbx.SummaryWriter(log_dir=self.log_path)

    def _close_writer(self):
        if self._writer:
            self._writer.close()


def main():
    """Main function."""
    params = parse_args()
    # All CLI values override the DEFAULT_* dictionaries declared above; edit those
    # dictionaries when you want to change the standard run configuration.
    lr_params = params["learning_rate"]
    cs_params = params["collect_stats"]
    params = params["other"]

    device = torch.device(params["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but no GPU is available. Set --device cpu to run on CPU.")
        device_index = device.index if device.index is not None else 0
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
        LOG.info("Using CUDA device %s (%s)", device, torch.cuda.get_device_name(device_index))
    else:
        raise RuntimeError("Only CUDA devices are supported by this training script. Use --device cuda (optionally with an index).")

    # ut.set_default_device("cuda")

    model = mm.DecoratorModel.load_from_file(params["input_model_path"])
    model.network.to(device)
    optimizer = torch.optim.Adam(model.network.parameters(), lr=lr_params["start"])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_params["step"], gamma=lr_params["gamma"])

    training_sets = load_sets(params["training_set_path"])
    validation_sets = []
    if params["collect_stats_frequency"] > 0:
        validation_sets = load_sets(cs_params["validation_set_path"])

    post_epoch_hook = TrainModelPostEpochHook(
        params["output_model_prefix_path"], params["epochs"], validation_sets, lr_scheduler,
        cs_params, lr_params, collect_stats_frequency=params["collect_stats_frequency"],
        save_frequency=params["save_every_n_epochs"], logger=LOG
    )

    epochs_it = ma.TrainModel(model, optimizer, training_sets, params["batch_size"], params["clip_gradients"],
                              params["epochs"], post_epoch_hook, logger=LOG).run()

    for num, (total, epoch_it) in enumerate(epochs_it):
        for _ in ul.progress_bar(epoch_it, total=total, desc="#{}".format(num)):
            pass  # we could do sth in here, but not needed :)


def load_sets(set_path):
    file_paths = [set_path]
    if os.path.isdir(set_path):
        file_paths = sorted(glob.glob("{}/*.smi".format(set_path)))

    for path in it.cycle(file_paths):  # stores the path instead of the set
        yield list(uc.read_csv_file(path, num_fields=2))


SUBCATEGORIES = ["collect_stats", "learning_rate"]


def _merge_defaults(defaults, provided):
    merged = defaults.copy()
    for key, value in provided.items():
        if value is not None:
            merged[key] = value
    return merged


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser(
        description="Train a model on a SMILES file.")

    _add_base_args(parser)
    _add_lr_args(parser)
    csfm.add_stats_args(parser, with_prefix=True, with_required=False)

    args = {k: {} for k in ["other", *SUBCATEGORIES]}
    for arg, val in vars(parser.parse_args()).items():
        done = False
        for prefix in SUBCATEGORIES:
            if arg.startswith(prefix):
                arg_name = arg[len(prefix) + 1:]
                args[prefix][arg_name] = val
                done = True
        if not done:
            args["other"][arg] = val

    args["collect_stats"] = _merge_defaults(DEFAULT_COLLECT_STATS_PARAMS, args["collect_stats"])
    args["learning_rate"] = _merge_defaults(DEFAULT_LEARNING_RATE_PARAMS, args["learning_rate"])
    args["other"] = _merge_defaults(DEFAULT_OTHER_PARAMS, args["other"])

    frequency = args["collect_stats"].pop("frequency", DEFAULT_COLLECT_STATS_PARAMS["frequency"])
    args["other"]["collect_stats_frequency"] = frequency

    #输出所有参数
    for category, cat_args in args.items():
        for key, value in cat_args.items():
            LOG.info("Parameter %s (%s): %s", key, category, value)
    return args


def _add_lr_args(parser):
    parser.add_argument("--learning-rate-start", "--lrs",
                        help="Starting learning rate for training. [DEFAULT: {start}]".format(
                            start=DEFAULT_LEARNING_RATE_PARAMS["start"]),
                        type=float, default=DEFAULT_LEARNING_RATE_PARAMS["start"])
    parser.add_argument("--learning-rate-min", "--lrmin",
                        help="Minimum learning rate, when reached the training stops. [DEFAULT: {min}]".format(
                            min=DEFAULT_LEARNING_RATE_PARAMS["min"]),
                        type=float, default=DEFAULT_LEARNING_RATE_PARAMS["min"])
    parser.add_argument("--learning-rate-gamma", "--lrg",
                        help="Ratio which the learning change is changed. [DEFAULT: {gamma}]".format(
                            gamma=DEFAULT_LEARNING_RATE_PARAMS["gamma"]),
                        type=float, default=DEFAULT_LEARNING_RATE_PARAMS["gamma"])
    parser.add_argument("--learning-rate-step", "--lrt",
                        help="Number of epochs until the learning rate changes. [DEFAULT: {step}]".format(
                            step=DEFAULT_LEARNING_RATE_PARAMS["step"]),
                        type=int, default=DEFAULT_LEARNING_RATE_PARAMS["step"])


def _add_base_args(parser):
    parser.add_argument("--input-model-path", "-i",
                        help="Input model file. [DEFAULT: {path}]".format(
                            path=DEFAULT_OTHER_PARAMS["input_model_path"]),
                        type=str, default=DEFAULT_OTHER_PARAMS["input_model_path"], required=False)
    parser.add_argument("--output-model-prefix-path", "-o",
                        help="Prefix to the output model (may have the epoch appended). [DEFAULT: {path}]".format(
                            path=DEFAULT_OTHER_PARAMS["output_model_prefix_path"]),
                        type=str, default=DEFAULT_OTHER_PARAMS["output_model_prefix_path"], required=False)
    parser.add_argument("--training-set-path", "-s",
                        help="Path to a file with (scaffold, decoration) tuples or a directory with many of these files to be used as training set. [DEFAULT: {path}]".format(
                            path=DEFAULT_OTHER_PARAMS["training_set_path"]),
                        type=str, default=DEFAULT_OTHER_PARAMS["training_set_path"], required=False)
    parser.add_argument("--save-every-n-epochs", "--sen",
                        help="Save the model after n epochs. [DEFAULT: {val}]".format(
                            val=DEFAULT_OTHER_PARAMS["save_every_n_epochs"]),
                        type=int, default=DEFAULT_OTHER_PARAMS["save_every_n_epochs"])
    parser.add_argument("--epochs", "-e",
                        help="Number of epochs to train. [DEFAULT: {val}]".format(
                            val=DEFAULT_OTHER_PARAMS["epochs"]),
                        type=int, default=DEFAULT_OTHER_PARAMS["epochs"])
    parser.add_argument("--batch-size", "-b",
                        help="Number of molecules processed per batch. [DEFAULT: {val}]".format(
                            val=DEFAULT_OTHER_PARAMS["batch_size"]),
                        type=int, default=DEFAULT_OTHER_PARAMS["batch_size"])
    parser.add_argument("--clip-gradients",
                        help="Clip gradients to a given norm. [DEFAULT: {val}]".format(
                            val=DEFAULT_OTHER_PARAMS["clip_gradients"]),
                        type=float, default=DEFAULT_OTHER_PARAMS["clip_gradients"])
    parser.add_argument("--device", "-D",
                        help="Torch device to run training on. [DEFAULT: {val}]".format(
                            val=DEFAULT_OTHER_PARAMS["device"]),
                        type=str, default=DEFAULT_OTHER_PARAMS["device"], required=False)
    parser.add_argument("--collect-stats-frequency", "--csf",
                        help="Collect statistics every n epochs. [DEFAULT: {val}]".format(
                            val=DEFAULT_COLLECT_STATS_PARAMS["frequency"]),
                        type=int, default=DEFAULT_COLLECT_STATS_PARAMS["frequency"])


if __name__ == "__main__":
    LOG = ul.get_logger(name="train_model")
    main()
