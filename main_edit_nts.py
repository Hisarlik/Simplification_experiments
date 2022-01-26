import sys

sys.dont_write_bytecode = True
from pathlib import Path

from src.editnts.data import preprocess
from src.utils import logging_module
from src.editnts.constants import TRAIN_ORIGINAL_DATA_PATH,TRAIN_SIMPLE_DATA_PATH

logger = logging_module.get_logger(__name__)

def main():
    logger.info("Init editNTS")

    df = preprocess.preprocess_raw_data(
        Path(TRAIN_ORIGINAL_DATA_PATH), Path(TRAIN_SIMPLE_DATA_PATH)
    )
    print(df)


if __name__ == "__main__":
    main()
