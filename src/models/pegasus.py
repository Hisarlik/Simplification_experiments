from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer
from constants import DEVICE

TRAIN_ORIGINAL_DATA_PATH = "resources/wikismall/PWKP_108016.tag.80.aner.ori.train.src"
TRAIN_SIMPLE_DATA_PATH = "resources/wikismall/PWKP_108016.tag.80.aner.ori.train.dst"


class PegasusSumModel(object):

    def __init__(self, tokenizer,  args=None):

        self.pretrained_model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-cnn_dailymail").to(DEVICE)
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorForSeq2Seq(tokenizer, model=self.pretrained_model)
        self.training_args = TrainingArguments(
            output_dir='pegasus-samsum',
            num_train_epochs=1,
            warmup_steps=500,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            weight_decay=0.01,
            logging_steps=10,
            evaluation_strategy='steps',
            eval_steps=500,
            save_steps=1e6,
            gradient_accumulation_steps=16)


    def train(self, **data):

        trainer = Trainer(model=self.pretrained_model,
                          args=self.training_args,
                          tokenizer=self.tokenizer,
                          data_collator=self.data_collator,
                          train_dataset=data['train'],
                          eval_dataset=data['valid'])

        trainer.train()