# -- fix path -- Source: https://github.com/KimChengSHEANG/TS_T5
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --
from typing import Dict
from source.experiments import ExperimentManager
from conf import WIKILARGE_DATASET, TURKCORPUS_DATASET, ASSET_DATASET, DEVICE
from source.evaluation import evaluate

def main(model_hyperparameters: Dict,
         features: Dict
         ):
    """
    :param model_hyperparameters: Language model parameters. e.g. : selected model, learning rate, ...
    :param features: A Dict with simplification features e.g. : WordLengthRatio, WordRankRatio, ...
    :return:
    """

    experiment = ExperimentManager.create_experiment(
        model_hyperparameters,
        features
    )

    experiment.start()
    ExperimentManager.save_experiment(experiment)


if __name__ == "__main__":

    features_6 = dict(
        WordLengthRatio=dict(target_ratio=0.75),
        CharLengthRatio=dict(target_ratio=0.6),
        LevenshteinRatio=dict(target_ratio=0.6),
        DependencyTreeDepthRatio=dict(target_ratio=0.95),
        WordRankRatio = dict(target_ratio=0.8),
        LMFillMaskRatio=dict(target_ratio=0.75)
    )



    config = dict(
        model_name='t5-small',
        dataset_path=WIKILARGE_DATASET,
        number_epochs=10,
        max_seq_length=256,
        learning_rate=3e-4,
        weight_decay=0.1,
        adam_epsilon=1e-8,
        warmup_steps=5,
        train_batch_size=6,
        valid_batch_size=6,
        custom_loss=False,
        gradient_accumulation_steps=1,
        accelerator="gpu",
        fp_16=False,
        opt_level='O1',
        max_grad_norm=1.0,
        seed=42,
        nb_sanity_val_steps=0,
        train_sample_size=1,
        valid_sample_size=1,
        device=DEVICE                   # "cuda" or "cpu"
    )

    print("Con features", len(features_6.keys()))
    main(config, features_6)




