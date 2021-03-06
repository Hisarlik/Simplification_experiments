# -- fix path -- Source: https://github.com/KimChengSHEANG/TS_T5
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --
import torch
from src.utils import logging_module
from src.data.preprocessing.t5_pre import T5_Preprocessing
from src.models.t5_simplification import TrainerT5
from constants import WIKILARGE_CHUNK_TRAIN_ORIGINAL_DATA_PATH, \
    WIKILARGE_CHUNK_VALID_ORIGINAL_DATA_PATH, \
    WIKILARGE_CHUNK_TEST_ORIGINAL_DATA_PATH, \
    WIKILARGE_CHUNK_TRAIN_SIMPLE_DATA_PATH, \
    WIKILARGE_CHUNK_VALID_SIMPLE_DATA_PATH, \
    WIKILARGE_CHUNK_TEST_SIMPLE_DATA_PATH, \
    MODEL_CKPT, PREPROCESSED_DIR, OUTPUT_DIR, DEVICE


logger = logging_module.get_logger(__name__)

if __name__ == "__main__":


    model_hyperparameters = dict(
        model_name='t5-small',
        learning_rate=3e-4,
        weight_decay=0.1,
        adam_epsilon=1e-8,
        warmup_steps=5,
        train_batch_size=6,
        valid_batch_size=6,
        num_train_epochs=5,
        custom_loss=False,
        gradient_accumulation_steps=1,  # 16
        n_gpu=torch.cuda.device_count(),
        fp_16=False,
        opt_level='O1',
        max_grad_norm=1.0,
        seed=12,
        nb_sanity_val_steps=0,
        train_sample_size=1,
        valid_sample_size=1,
        output_dir=OUTPUT_DIR,
        device=DEVICE
    )


    files_path = {"train_original_text_path": WIKILARGE_CHUNK_TRAIN_ORIGINAL_DATA_PATH,
                  "valid_original_text_path": WIKILARGE_CHUNK_VALID_ORIGINAL_DATA_PATH,
                  "test_original_text_path": WIKILARGE_CHUNK_TEST_ORIGINAL_DATA_PATH,
                  "train_simple_text_path": WIKILARGE_CHUNK_TRAIN_SIMPLE_DATA_PATH,
                  "valid_simple_text_path": WIKILARGE_CHUNK_VALID_SIMPLE_DATA_PATH,
                  "test_simple_text_path": WIKILARGE_CHUNK_TEST_SIMPLE_DATA_PATH
                  }

    features = {
        'WordLengthRatio': {'target_ratio': 0.8},
        'CharLengthRatio': {'target_ratio': 0.8},
        'LevenshteinRatio': {'target_ratio': 0.8},
        'DependencyTreeDepthRatio': {'target_ratio': 0.8},
        'WordRankRatio': {'target_ratio': 0.8}
    }

    conf = {
                "model_hyperparameters": model_hyperparameters,
                "model_ckpt" : MODEL_CKPT,
                "type_dataset": "huggingface",
                "files_path": files_path,
                "features": features,
                "preprocessed_data_path": PREPROCESSED_DIR / "wikilarge_chunk",
                "max_seq_length" : 32,
    }

    t5_pre = T5_Preprocessing(**conf)
    #t5_pre.preprocess()
    trainer = TrainerT5(t5_pre, model_hyperparameters)
    trainer.train()