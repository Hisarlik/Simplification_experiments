# -- fix path --
from pathlib import Path
import sys
from typing import Union

sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

from source.utils import logging_module

import optuna
from conf import WIKILARGE_CHUNK_DATASET, TURKCORPUS_DATASET, WIKILARGE_DATASET, SIMPLETEXT_DATASET
from source.experiments import ExperimentManager
from source.evaluation import evaluate

logger = logging_module.get_logger(__name__)


def objective(trial: optuna.trial.Trial, experiment_id:Union[str, None], dataset: Path) -> float:
    features = dict(
        WordLengthRatio=dict(target_ratio=trial.suggest_float('WordRatio', 0.7, 1.3, step=0.05)),
        CharLengthRatio=dict(target_ratio=trial.suggest_float('CharRatio', 0.7, 1.3, step=0.05)),
        LevenshteinRatio=dict(target_ratio=trial.suggest_float('LevenshteinRatio', 0.6, 1., step=0.05)),
        DependencyTreeDepthRatio=dict(target_ratio=trial.suggest_float('DepthTreeRatio', 0.8, 1.5, step=0.05)),
        WordRankRatio=dict(target_ratio=trial.suggest_float('WordRankRatio', 0.5, 1, step=0.05)),
        LMFillMaskRatio=dict(target_ratio=trial.suggest_float('LMFillMaskRatio', 0.7, 1.3, step=0.05)))

    experiment = ExperimentManager.load_experiment(experiment_id)
    #experiment.experiment_path = Path("/home/antonio/PycharmProjects/Simplification_experiments/resources/experiments/20220907200906")
    result = evaluate(experiment, dataset, features, "validation")
    return result


if __name__ == '__main__':

    # Change experiment_id value or None to evaluate last trained model.
    # Select dataset for finetuning it.
    expe_id = None
    dataset = TURKCORPUS_DATASET
    trials = 200


    func = lambda trial: objective(trial, expe_id, dataset)
    study = optuna.create_study(study_name='Tokens_study', direction="maximize")
    study.optimize(func, n_trials=trials)

    logger.info("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    logger.info("  Value: {}".format(trial.value))

    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))
