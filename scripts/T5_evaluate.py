# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

from conf import WIKILARGE_CHUNK_DATASET, TURKCORPUS_DATASET, WIKILARGE_DATASET, SIMPLETEXT_DATASET, ASSET_DATASET
from source.experiments import ExperimentManager
from source.evaluation import evaluate

if __name__ == "__main__":
    features = dict(
        WordLengthRatio=dict(target_ratio=1.05),
        CharLengthRatio=dict(target_ratio=0.9),
        LevenshteinRatio=dict(target_ratio=0.8),
        DependencyTreeDepthRatio=dict(target_ratio=0.9),
        WordRankRatio = dict(target_ratio=0.7),
        LMFillMaskRatio=dict(target_ratio=1.25)
    )
    # Select experiment_id value or put None to evaluate last trained model.
    experiment_id = None
    dataset = TURKCORPUS_DATASET
    split = "validation"

    experiment = ExperimentManager.load_experiment(experiment_id)
    #experiment.experiment_path = Path("/home/antonio/PycharmProjects/Simplification_experiments/resources/experiments/20221003000114")
    evaluate(experiment, dataset, features, split)
