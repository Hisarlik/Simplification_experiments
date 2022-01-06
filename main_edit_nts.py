import sys

sys.dont_write_bytecode = True
from pathlib import Path

from src.data import storage
from src.utils import logging_module

logger = logging_module.get_logger(__name__)


path = "data/wikismall/PWKP_108016.tag.80.aner.ori.train.src"


def main():
    logger.info("Working main editNTS")
    pathfile = Path(path)
    df = storage.load_dataset_into_pandas(pathfile, pathfile, "\t")
    print(df.head())


if __name__ == "__main__":
    main()
