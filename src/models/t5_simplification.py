from constants import DEVICE
from src.utils import logging_module
import pytorch_lightning as pl
from pytorch_lightning.trainer import seed_everything
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    get_linear_schedule_with_warmup, AutoConfig, AutoModel
)
import os

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

    def __init__(self,  preprocess, args=None):

        self.checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.get('output_dir'),
            filename="checkpoint-{epoch}",
            monitor="val_loss",
            verbose=True,
            mode="min",
            save_top_k=5
        )
        train_params = dict(
            accelerator="cpu",
            accumulate_grad_batches=args.get('gradient_accumulation_steps'),
            gpus=args.get('n_gpu'),
            max_epochs=args.get('num_train_epochs'),
            # early_stop_callback=False,
            precision=16 if args.get('fp_16') else 32,
            callbacks=[LoggingCallback(), self.checkpoint_callback],
            num_sanity_val_steps=args.get('nb_sanity_val_steps'),
            progress_bar_refresh_rate=1,
        )

        self.trainer = pl.Trainer(**train_params)
        self.model = T5SimplificationModel(preprocess, **args)

    def train(self):

        self.trainer.fit(self.model)


class T5SimplificationModel(pl.LightningModule):

    def __init__(self, preprocess, **kwarg):
        super(T5SimplificationModel, self).__init__()
        self.save_hyperparameters()
        self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name).to(self.hparams.device)
        self.tokenizer = T5TokenizerFast.from_pretrained(self.hparams.model_name)

    def is_logger(self):
        return self.trainer.proc_rank <= 0


    def forward(
        self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            #decoder_input_ids=labels,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )
        return outputs

    def _step(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100
        self.opt.zero_grad()

        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
            decoder_attention_mask=batch['target_mask'],
        )
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                               lr=self.hparams.learning_rate,
                               eps=self.hparams.adam_epsilon)

        self.opt = optimizer
        return optimizer

    def optimizer_step(self, epoch=None, batch_idx=None, optimizer=None, optimizer_idx=None, optimizer_closure=None,
                       on_tpu=None, using_native_amp=None, using_lbfgs=None):
        optimizer.step(closure=optimizer_closure)

        optimizer.zero_grad()
        self.lr_scheduler.step()


    def train_dataloader(self):
        path = self.hparams.preprocess.kwargs.get("preprocessed_data_path")
        train_dataset = self.hparams.preprocess.load_from_disk(path)
        train_dataset = self.hparams.preprocess.tokenize(train_dataset["train"])
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size)
        return dataloader

    def val_dataloader(self):
        path = self.hparams.preprocess.kwargs.get("preprocessed_data_path")
        valid_dataset = self.hparams.preprocess.load_from_disk(path)
        valid_dataset = self.hparams.preprocess.tokenize(valid_dataset["valid"])
        dataloader = DataLoader(valid_dataset, batch_size=self.hparams.valid_batch_size)
        return dataloader

    def test_dataloader(self):
        pass





