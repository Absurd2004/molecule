import random
from pathlib import Path

import numpy as np

import torch
import torch.utils.data as tud
import torch.nn.utils as tnnu

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("RDKit is required for molecule visualization; install via 'conda install -c rdkit rdkit'.") from exc

import models.dataset as md
import utils.scaffold as usc


class Action:
    def __init__(self, logger=None):
        """
        (Abstract) Initializes an action.
        :param logger: An optional logger instance.
        """
        self.logger = logger

    def _log(self, level, msg, *args):
        """
        Logs a message with the class logger.
        :param level: Log level.
        :param msg: Message to log.
        :param *args: The arguments to escape.
        :return:
        """
        if self.logger:
            getattr(self.logger, level)(msg, *args)


class TrainModelPostEpochHook(Action):

    def __init__(self, logger=None):
        """
        Initializes a training hook that runs after every epoch.
        This hook enables to save the model, change LR, etc. during training.
        :return:
        """
        Action.__init__(self, logger)

    def run(self, model, training_set, epoch):  # pylint: disable=unused-argument
        """
        Performs the post-epoch hook. Notice that model should be modified in-place.
        :param model: Model instance trained up to that epoch.
        :param training_set: List of SMILES used as the training set.
        :param epoch: Epoch number (for logging purposes).
        :return: Boolean that indicates whether the training should continue or not.
        """
        return True  # simply does nothing...


class TrainModel(Action):

    def __init__(self, model, optimizer, training_sets, batch_size, clip_gradient,
                 epochs, post_epoch_hook=None, logger=None):
        """
        Initializes the training of an epoch.
        : param model: A model instance, not loaded in sampling mode.
        : param optimizer: The optimizer instance already initialized on the model.
        : param training_sets: An iterator with all the training sets (scaffold, decoration) pairs.
        : param batch_size: Batch size to use.
        : param clip_gradient: Clip the gradients after each backpropagation.
        : return:
        """
        Action.__init__(self, logger)

        self.model = model
        self.optimizer = optimizer
        self.training_sets = training_sets
        self.batch_size = batch_size
        self.epochs = epochs
        self.clip_gradient = clip_gradient

        if not post_epoch_hook:
            self.post_epoch_hook = TrainModelPostEpochHook(logger=self.logger)
        else:
            self.post_epoch_hook = post_epoch_hook

    def run(self):
        """
        Performs a training epoch with the parameters used in the constructor.
        :return: An iterator of (total_batches, epoch_iterator), where the epoch iterator
                  returns the loss function at each batch in the epoch.
        """
        print(f"start training for {self.epochs} epochs")
        for epoch, training_set in zip(range(1, self.epochs + 1), self.training_sets):
            dataloader = self._initialize_dataloader(training_set)
            epoch_iterator = self._epoch_iterator(dataloader)
            yield len(dataloader), epoch_iterator

            self.model.set_mode("eval")
            post_epoch_status = self.post_epoch_hook.run(self.model, training_set, epoch)
            self.model.set_mode("train")

            if not post_epoch_status:
                break

    def _epoch_iterator(self, dataloader):
        for scaffold_batch, decorator_batch in dataloader:
            loss = self.model.likelihood(*scaffold_batch, *decorator_batch).mean()
            #print(f"loss: {loss}")

            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_gradient > 0:
                tnnu.clip_grad_norm_(self.model.network.parameters(), self.clip_gradient)

            self.optimizer.step()

            yield loss

    def _initialize_dataloader(self, training_set):
        dataset = md.DecoratorDataset(training_set, vocabulary=self.model.vocabulary)
        return tud.DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                              collate_fn=md.DecoratorDataset.collate_fn, drop_last=True)


class CollectStatsFromModel(Action):
    """Collect and persist evaluation metrics for a decorator model."""

    def __init__(self, model, epoch, training_set, validation_set, sample_size,
                 output_dir, decoration_type="multi", other_values=None, logger=None,
                 max_mols_per_grid=0, individual_image_size=512):
        Action.__init__(self, logger)
        self.model = model
        self.epoch = epoch
        self.sample_size = sample_size
        self.training_set = training_set
        self.validation_set = validation_set
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.other_values = other_values or {}
        self.decoration_type = decoration_type
        self.max_mols_per_grid = max(0, int(max_mols_per_grid or 0))
        if isinstance(individual_image_size, (list, tuple)) and len(individual_image_size) == 2:
            width, height = individual_image_size
        else:
            width = height = int(individual_image_size)
        self.individual_image_size = (max(1, int(width)), max(1, int(height)))

        self.data = {}

        self._calc_nlls_action = CalculateNLLsFromModel(self.model, 128, logger=self.logger)
        self._sample_model_action = SampleModel(self.model, 128, logger=self.logger)

    @torch.no_grad()
    def run(self):
        #self._log("info", "Collecting data for epoch %s", self.epoch)
        print(f"Collecting data for epoch {self.epoch}")
        self.data = {"epoch": self.epoch}

        if isinstance(self.validation_set, list):
            validation_set_full = self.validation_set
        else:
            validation_set_full = list(self.validation_set)
            self.validation_set = validation_set_full
        #self._log("debug", "Validation set size: %d", len(validation_set_full))
        print(f"Validation set size: {len(validation_set_full)}")

        validation_nlls = list(self._calc_nlls_action.run(validation_set_full))
        if validation_nlls:
            validation_nlls_arr = np.array(validation_nlls, dtype=np.float32)
            self.data["validation_nll_mean"] = float(validation_nlls_arr.mean())
            self.data["validation_nll_count"] = int(len(validation_nlls_arr))
        else:
            self.data["validation_nll_mean"] = float("nan")
            self.data["validation_nll_count"] = 0

        scaffolds_for_sampling = [sc for sc, _ in validation_set_full]
        if not scaffolds_for_sampling:
            #self._log("warning", "Validation set is empty; skipping molecule generation")
            print("Validation set is empty; skipping molecule generation")
            return self._merge_other_values()

        if self.sample_size and self.sample_size < len(scaffolds_for_sampling):
            scaffolds_for_sampling = scaffolds_for_sampling[:self.sample_size]

        sampled_mols, _ = self._sample_decorations(scaffolds_for_sampling)
        total_requested = len(scaffolds_for_sampling)
        valid_generated = len(sampled_mols)
        success_ratio = (valid_generated / total_requested) if total_requested else 0.0

        self.data["generated_molecule_total"] = total_requested
        self.data["generated_molecule_valid"] = valid_generated
        self.data["generated_molecule_valid_ratio"] = success_ratio

        image_payload = self._save_molecule_grid(sampled_mols)
        if image_payload:
            grid_paths = image_payload.get("grid", [])
            if grid_paths:
                self.data["generated_molecule_image_path"] = str(grid_paths[0])
                self.data["generated_molecule_image_paths"] = "\n".join(str(path) for path in grid_paths)

            individual_paths = image_payload.get("individual", [])
            if individual_paths:
                self.data["generated_molecule_individual_paths"] = "\n".join(str(path) for path in individual_paths)
                self.data["generated_molecule_individual_count"] = len(individual_paths)
                if not grid_paths:
                    self.data["generated_molecule_image_path"] = str(individual_paths[0])
                    self.data["generated_molecule_image_paths"] = "\n".join(str(path) for path in individual_paths)

            smiles_file = image_payload.get("smiles")
            if smiles_file:
                self.data["generated_molecule_smiles_path"] = str(smiles_file)

        return self._merge_other_values()

    def _merge_other_values(self):
        if self.other_values:
            self.data.update(self.other_values)
        return self.data

    def _sample_decorations(self, scaffold_list):
        mol_smis = []
        nlls = []
        for scaff, decoration, nll in self._sample_model_action.run(scaffold_list):
            #mol = usc.join_first_attachment(scaff, decoration)
            mol = usc.join_joined_attachments(scaff, decoration)
            if mol:
                mol_smi = Chem.MolToSmiles(mol, isomericSmiles=False)
                mol_smis.append(mol_smi)
            nlls.append(nll)
        return mol_smis, np.array(nlls, dtype=np.float32) if nlls else np.array([], dtype=np.float32)

    def _save_molecule_grid(self, mol_smiles):
        if not mol_smiles:
            return None

        mol_records = []
        for sm in mol_smiles:
            mol = Chem.MolFromSmiles(sm)
            if mol is not None:
                mol_records.append((sm, mol))

        if not mol_records:
            self._log("warning", "All generated molecules failed to parse; skipping image save")
            return None

        epoch_dir = self.output_dir / f"epoch_{self.epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        smiles_path = epoch_dir / "generated_molecules.smi"
        with smiles_path.open("w", encoding="utf-8") as handle:
            for sm, _ in mol_records:
                handle.write(f"{sm}\n")

        grid_paths = []
        if self.max_mols_per_grid > 0:
            chunk_size = self.max_mols_per_grid
            mol_objects = [mol for _, mol in mol_records]

            def _chunks(sequence, size):
                for start in range(0, len(sequence), size):
                    yield sequence[start:start + size]

            for chunk_index, chunk in enumerate(_chunks(mol_objects, chunk_size), start=1):
                mols_per_row = min(4, max(1, len(chunk)))
                image = Draw.MolsToGridImage(chunk, molsPerRow=mols_per_row, subImgSize=(250, 250))
                grid_path = epoch_dir / f"generated_molecules_{chunk_index:03d}.png"
                image.save(str(grid_path))
                grid_paths.append(grid_path)

        individual_dir = epoch_dir / "individual"
        individual_dir.mkdir(parents=True, exist_ok=True)

        individual_paths = []
        for idx, (_, mol) in enumerate(mol_records, start=1):
            indiv_path = individual_dir / f"mol_{idx:05d}.png"
            self._draw_molecule_full_frame(mol, indiv_path)
            individual_paths.append(indiv_path)

        return {
            "grid": grid_paths,
            "individual": individual_paths,
            "smiles": smiles_path,
        }

    def _draw_molecule_full_frame(self, mol, output_path):
        width, height = self.individual_image_size
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        drawer.drawOptions().padding = 0.0
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        output_path.write_bytes(drawer.GetDrawingText())


class SampleModel(Action):

    def __init__(self, model, batch_size, logger=None):
        """
        Creates an instance of SampleModel.
        :params model: A model instance (better in sampling mode).
        :params batch_size: Batch size to use.
        :return:
        """
        Action.__init__(self, logger)
        self.model = model
        self.batch_size = batch_size

    def run(self, scaffold_list):
        """
        Samples the model for the given number of SMILES.
        :params scaffold_list: A list of scaffold SMILES.
        :return: An iterator with each of the batches sampled in (scaffold, decoration, nll) triplets.
        """
        dataset = md.Dataset(scaffold_list, self.model.vocabulary.scaffold_vocabulary,
                             self.model.vocabulary.scaffold_tokenizer)
        dataloader = tud.DataLoader(dataset, batch_size=self.batch_size,
                                    shuffle=False, collate_fn=md.Dataset.collate_fn)
        for batch in dataloader:
            for scaff, dec, nll in self.model.sample_decorations(*batch):
                yield scaff, dec, nll


class CalculateNLLsFromModel(Action):

    def __init__(self, model, batch_size, with_attention_weights=False, logger=None):
        """
        Creates an instance of CalculateNLLsFromModel.
        :param model: A model instance.
        :param batch_size: Batch size to use.
        :return:
        """
        Action.__init__(self, logger)
        self.model = model
        self.batch_size = batch_size
        self.with_attention_weights = with_attention_weights

    def run(self, scaffold_decoration_list):
        """
        Calculates the NLL for a set of SMILES strings.
        :param scaffold_decoration_list: List with pairs of (scaffold, decoration) SMILES.
        :return: An iterator with each NLLs in the same order as the list.
        """
        dataset = md.DecoratorDataset(scaffold_decoration_list, self.model.vocabulary)
        dataloader = tud.DataLoader(dataset, batch_size=self.batch_size, collate_fn=md.DecoratorDataset.collate_fn,
                                    shuffle=False)
        for scaffold_batch, decorator_batch in dataloader:
            ll_data = self.model.likelihood(*scaffold_batch, *decorator_batch,
                                            with_attention_weights=self.with_attention_weights)
            if self.with_attention_weights:
                data = zip(*[d.data.cpu().numpy() for d in ll_data])
            else:
                data = ll_data.data.cpu().numpy()
            for ll_d in data:
                yield ll_d
