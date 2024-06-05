# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

from conf import WIKILARGE_CHUNK_DATASET, TURKCORPUS_DATASET, WIKILARGE_DATASET, SIMPLETEXT_DATASET, ASSET_DATASET, DIARIOMADRID_DATASET
from source.experiments import ExperimentManager
from source.evaluation import evaluate

if __name__ == "__main__":
    features = dict(
        WordLengthRatio=dict(target_ratio=1.3),
        CharLengthRatio=dict(target_ratio=0.85),
        LevenshteinRatio=dict(target_ratio=0.65),
        DependencyTreeDepthRatio=dict(target_ratio=0.9),
        WordRankRatio = dict(target_ratio=0.6),
        LMFillMaskRatio=dict(target_ratio=1.15)
    )
    # Select experiment_id value or put None to evaluate last trained model.
    experiment_id = "20220418231244"
    dataset = DIARIOMADRID_DATASET
    split = "test"

    experiment = ExperimentManager.load_experiment(experiment_id)
    experiment.experiment_path = Path("/home/antonio/PycharmProjects/Simplification_experiments/resources/experiments/20220418231244")
    evaluate(experiment, dataset, features, split)
