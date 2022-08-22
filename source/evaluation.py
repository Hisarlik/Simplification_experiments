# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --
from typing import Dict
from pathlib import Path
from source.experiments import Experiment
from source.utils import logging_module

logger = logging_module.get_logger(__name__)

def evaluate(experiment: Experiment,
             dataset: Path,
             features: Dict,
             split: str,
             metrics : bool = True):

    logger.info(f"Evaluation Dataset: {dataset.name} "
                f"with experiment: {experiment.experiment_path.name} "
                f"and split: {split}")

    dm = experiment.create_and_setup_data_module(dataset, features, "test", split)
    trainer = experiment.create_trainer()
    model = experiment.load_best_model()
    trainer.test(model, datamodule=dm)
    if metrics:
        return experiment.get_metrics(model, dm, split)
