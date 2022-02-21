from transformers import T5ForConditionalGeneration, T5TokenizerFast

from constants import DEVICE
from src.utils import logging_module
import pytorch_lightning as pl
from pytorch_lightning.trainer import seed_everything
import os
import argparse

logger = logging_module.get_logger(__name__)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))

class TrainerT5(object):

    def __init__(self,  hyperparameters=None):
        args = argparse.Namespace(**hyperparameters)

        self.checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir,
            filename="checkpoint-{epoch}",
            monitor="val_loss",
            verbose=True,
            mode="min",
            save_top_k=5
        )
        train_params = dict(
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gpus=args.n_gpu,
            max_epochs=args.num_train_epochs,
            # early_stop_callback=False,
            precision=16 if args.fp_16 else 32,
            #amp_level=args.opt_level,
            # gradient_clip_val=args.max_grad_norm,
            # checkpoint_callback=checkpoint_callback,
            callbacks=[LoggingCallback(), self.checkpoint_callback],
            # logger=TensorBoardLogger(f'{args.output_dir}/logs'),
            num_sanity_val_steps=args.nb_sanity_val_steps,  # skip sanity check to save time for debugging purpose
            # plugins='ddp_sharded',
            progress_bar_refresh_rate=1,

        )
        del hyperparameters['opt_level']
        del hyperparameters['fp_16']
        self.trainer = pl.Trainer(**train_params)
        self.model = T5SimplificationModel(**hyperparameters)

    def train(self, model):

        self.trainer.fit(model)




class T5SimplificationModel(pl.LightningModule):

    def __init__(self, *args, **kwarg):
        super(T5SimplificationModel, self).__init__()
        self.save_hyperparameters()
        print(self.hparams)
        self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name).to(self.hparams.device)
        self.tokenizer = T5TokenizerFast.from_pretrained(self.hparams.model_name)


