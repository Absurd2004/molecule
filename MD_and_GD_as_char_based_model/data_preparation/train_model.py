#!/usr/bin/env python
#  coding=utf-8

"""
Script to train a model
"""

import argparse
import os.path
import glob
import itertools as it
import math
from datetime import datetime
from pathlib import Path

import torch
from importlib import import_module


def _resolve_tqdm():
    try:
        return import_module("tqdm").tqdm
    except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
        def _passthrough(iterable, **kwargs):
            return iterable

        return _passthrough


tqdm = _resolve_tqdm()

import collect_stats_from_model as csfm

import models.model as mm
import models.actions as ma

import utils.chem as uc
import utils.log as ul

try:
    import wandb
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("Weights & Biases (wandb) is required for logging; install it via 'pip install wandb'.") from exc


class TrainModelPostEpochHook(ma.TrainModelPostEpochHook):
    def __init__(self, output_prefix_path, epochs, validation_sets, lr_scheduler, collect_stats_params,
                 lr_params, collect_stats_frequency, save_frequency, logger=None, global_step_fn=None):
        ma.TrainModelPostEpochHook.__init__(self, logger)

        self.validation_sets = validation_sets
        self.lr_scheduler = lr_scheduler

        self.output_prefix_path = output_prefix_path
        self.save_frequency = save_frequency
        self.epochs = epochs

        self.collect_stats_params = collect_stats_params
        self.collect_stats_frequency = collect_stats_frequency

        self.lr_params = lr_params
        log_dir = Path(self.collect_stats_params.get("log_path", "./wandb_logs")).expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = log_dir
        self._global_step_fn = global_step_fn
        self.best_validation_loss = None
        self.best_validation_epoch = None
        self._epochs_without_improvement = 0
        self._patience = 6
        self._best_model_path = Path(self.output_prefix_path).with_name("best.pt")

    def run(self, model, training_set, epoch):
        patience_triggered = False
        if self.collect_stats_frequency > 0 and epoch % self.collect_stats_frequency == 0:
            validation_set = next(self.validation_sets)
            other_values = {"lr": self.get_lr()}

            stats = ma.CollectStatsFromModel(
                model=model,
                epoch=epoch,
                training_set=training_set,
                validation_set=validation_set,
                sample_size=self.collect_stats_params["sample_size"],
                output_dir=self.output_dir,
                other_values=other_values,
                logger=self.logger,
                max_mols_per_grid=self.collect_stats_params.get("max_mols_per_grid", 0),
                individual_image_size=self.collect_stats_params.get("individual_image_size", 512)
            ).run()
            if stats:
                step_value = epoch
                if self._global_step_fn:
                    try:
                        step_value = self._global_step_fn()
                    except Exception:  # pragma: no cover - defensive fallback
                        step_value = epoch
                patience_triggered = self._update_validation_tracking(stats, model, epoch)
                wandb.log(stats, step=step_value)

        self.lr_scheduler.step(epoch=epoch)

        lr_reached_min = (self.get_lr() < self.lr_params["min"])
        if lr_reached_min or self.epochs == epoch \
                or (self.save_frequency > 0 and (epoch % self.save_frequency == 0)):
            model.save(self._model_path(epoch))

        return not lr_reached_min and not patience_triggered

    def get_lr(self):
        return self.lr_scheduler.optimizer.param_groups[0]["lr"]

    def _model_path(self, epoch):
        return "{}.{}".format(self.output_prefix_path, epoch)

    # writer management removed; logging handled via Weights & Biases

    def _update_validation_tracking(self, stats, model, epoch):
        val_loss = stats.get("validation_nll_mean")
        if val_loss is None or not math.isfinite(val_loss):
            return False

        if self.best_validation_loss is None or val_loss < self.best_validation_loss:
            self.best_validation_loss = val_loss
            self.best_validation_epoch = epoch
            self._epochs_without_improvement = 0
            self._best_model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(self._best_model_path))
            if self.logger:
                self.logger.info(
                    "New best validation loss %.6f at epoch %d; model saved to %s",
                    val_loss,
                    epoch,
                    self._best_model_path,
                )
            stats["best_validation_epoch"] = epoch
            stats["best_validation_loss"] = val_loss
            return False

        self._epochs_without_improvement += 1
        if self.best_validation_epoch is not None:
            stats.setdefault("best_validation_epoch", self.best_validation_epoch)
        if self.best_validation_loss is not None:
            stats.setdefault("best_validation_loss", self.best_validation_loss)
        if self._epochs_without_improvement >= self._patience:
            if self.logger:
                self.logger.info(
                    "Early stopping triggered after %d epochs without validation improvement", self._patience
                )
            return True
        return False


def main():
    """Main function."""
    params = parse_args()
    lr_params = params["learning_rate"]
    cs_params = params["collect_stats"]
    params = params["other"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = Path(params["output_model_prefix_path"]).expanduser()
    timestamp_parent = output_prefix.parent / timestamp
    timestamp_parent.mkdir(parents=True, exist_ok=True)
    prefix_basename = output_prefix.name if output_prefix.name else "model"
    timestamped_prefix = timestamp_parent / prefix_basename
    params["output_model_prefix_path"] = str(timestamped_prefix)
    LOG.info("Model checkpoints will be saved under %s", timestamped_prefix)

    images_root = Path("./images").expanduser()
    cs_params["log_path"] = str((images_root / timestamp).resolve())

    wandb_project = params.pop("wandb_project", None)
    wandb_run_name = params.pop("wandb_run_name", None)

    wandb_kwargs = {}
    if wandb_project:
        wandb_kwargs["project"] = wandb_project
    if wandb_run_name:
        wandb_kwargs["name"] = wandb_run_name

    wandb_config = {
        "learning_rate": lr_params,
        "collect_stats": cs_params,
        "training": {k: v for k, v in params.items() if k not in {"collect_stats_frequency"}}
    }

    wandb.init(**wandb_kwargs, config=wandb_config)

    # ut.set_default_device("cuda")

    model = mm.DecoratorModel.load_from_file(params["input_model_path"])
    optimizer = torch.optim.Adam(model.network.parameters(), lr=lr_params["start"])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_params["step"], gamma=lr_params["gamma"])

    training_sets = load_sets(params["training_set_path"])
    validation_sets = []
    if params["collect_stats_frequency"] > 0:
        validation_sets = load_sets(cs_params["validation_set_path"])

    global_step = 0

    post_epoch_hook = TrainModelPostEpochHook(
        params["output_model_prefix_path"], params["epochs"], validation_sets, lr_scheduler,
        cs_params, lr_params, collect_stats_frequency=params["collect_stats_frequency"],
        save_frequency=params["save_every_n_epochs"], logger=LOG,
        global_step_fn=lambda: global_step
    )

    epochs_it = ma.TrainModel(model, optimizer, training_sets, params["batch_size"], params["clip_gradients"],
                              params["epochs"], post_epoch_hook, logger=LOG).run()
    try:
        for epoch_num, (total, epoch_it) in enumerate(epochs_it, start=1):
            batch_bar = tqdm(epoch_it, total=total, desc=f"#{epoch_num}", unit="batch")
            for batch_idx, loss in enumerate(batch_bar):
                loss_value = float(loss.item()) if hasattr(loss, "item") else float(loss)
                batch_bar.set_postfix(loss=f"{loss_value:.4f}")
                wandb.log({
                    "train_batch_loss": loss_value,
                    "epoch": epoch_num,
                    "batch": batch_idx
                }, step=global_step)
                global_step += 1
    finally:
        wandb.finish()


def load_sets(set_path):
    """Load scaffold/decoration pairs from a file or directory with progress prints."""

    file_paths = [set_path]
    if os.path.isdir(set_path):
        file_paths = sorted(glob.glob("{}/*.smi".format(set_path)))
        print(f"Discovered {len(file_paths)} SMILES files in {set_path}")
    else:
        print(f"Preparing SMILES pairs from {set_path}")

    cached_sets = {}

    for path in it.cycle(file_paths):
        if path not in cached_sets:
            print(f"Loading scaffold/decoration pairs from {path}")
            pairs = []
            for row in tqdm(uc.read_csv_file(path, num_fields=2), desc=f"Reading {Path(path).name}", unit="pair"):
                pairs.append(row)
            print(f"Finished {path}: {len(pairs)} pairs loaded")
            cached_sets[path] = pairs
        yield cached_sets[path]


SUBCATEGORIES = ["collect_stats", "learning_rate"]


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

    # special case
    args["other"]["collect_stats_frequency"] = args["collect_stats"]["frequency"]
    del args["collect_stats"]["frequency"]

    return args


def _add_lr_args(parser):
    parser.add_argument("--learning-rate-start", "--lrs",
                        help="Starting learning rate for training. [DEFAULT: 1E-4]", type=float, default=1E-4)
    parser.add_argument("--learning-rate-min", "--lrmin",
                        help="Minimum learning rate, when reached the training stops. [DEFAULT: 1E-6]",
                        type=float, default=1E-6)
    parser.add_argument("--learning-rate-gamma", "--lrg",
                        help="Ratio which the learning change is changed. [DEFAULT: 0.95]", type=float, default=0.95)
    parser.add_argument("--learning-rate-step", "--lrt",
                        help="Number of epochs until the learning rate changes. [DEFAULT: 1]",
                        type=int, default=10)


def _add_base_args(parser):
    parser.add_argument("--input-model-path", "-i", help="Input model file", type=str, default="./pretrained_models/empty_model/decorator_model.pt")
    parser.add_argument("--output-model-prefix-path", "-o",
                        help="Prefix to the output model (may have the epoch appended).", type=str, default="./pretrained_models/decorator_model_epoch")
    parser.add_argument("--training-set-path", "-s", help="Path to a file with (scaffold, decoration) tuples \
        or a directory with many of these files to be used as training set.", type=str, default="./data/randomized_smiles")
    parser.add_argument("--save-every-n-epochs", "--sen",
                        help="Save the model after n epochs. [DEFAULT: 1]", type=int, default=1)
    parser.add_argument("--epochs", "-e", help="Number of epochs to train. [DEFAULT: 100]", type=int, default=10)
    parser.add_argument("--batch-size", "-b",
                        help="Number of molecules processed per batch. [DEFAULT: 128]", type=int, default=128)
    parser.add_argument("--clip-gradients",
                        help="Clip gradients to a given norm. [DEFAULT: 1.0]", type=float, default=1.0)
    parser.add_argument("--collect-stats-frequency", "--csf",
                        help="Collect statistics every n epochs. [DEFAULT: 0]", type=int, default=1)
    parser.add_argument("--wandb-project",
                        help="Weights & Biases project name used for logging. [DEFAULT: env/W&B default]",
                        type=str, default="transfer_learning")
    parser.add_argument("--wandb-run-name",
                        help="Optional custom run name for Weights & Biases logging.", type=str, default="transfer_learning_run")


if __name__ == "__main__":
    LOG = ul.get_logger(name="train_model")
    main()
