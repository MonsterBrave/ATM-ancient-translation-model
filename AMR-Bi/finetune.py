# from peft import LoraConfig, TaskType, get_peft_model

import sys 
sys.path.append('../common')
from datautils import get_biMerged_tokenized_dataset            # original dataset merged reversed dataset
from datautils import get_AplusM_to_MplusA_tokenized_dataset    # handle dataset to be modern+ancient to ancient+modern
from datautils import get_tokenized_dataset                     # unidirectional original dataset with special tokens
from datautils import get_unidirectional_tokenized_dataset      # unidirectional original dataset without special tokens
from config import get_default_args
from logging_utils import get_logger

from transformers import(
    BartTokenizer,
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)

logger = get_logger(__name__, "info")

def finetune():
    args = get_default_args()
    logger.info(args)

    set_seed(args.seed)

    tokenizer = BartTokenizer.from_pretrained(args.tokenizer_path, clean_up_tokenization_spaces=True)
    model = BartForConditionalGeneration.from_pretrained(args.model_path)

    train_dataset = get_tokenized_dataset(args, tokenizer, 'train')
    train_len = len(train_dataset)
    eval_dataset = get_tokenized_dataset(args, tokenizer, 'validation')

    collator = DataCollatorForSeq2Seq(tokenizer, model)

    # debug
    batch = collator([i for i in train_dataset.select(range(5))])
    logger.debug(batch)

    # lr = 3e-4 if args.peft else 5e-5
    lr = 5e-5
    eval_steps = int(train_len / args.batch_size * 0.2)

    training_args = TrainingArguments(
        output_dir=args.ft_output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        save_total_limit=2,
        load_best_model_at_end=True,
        save_strategy="steps",
        save_steps=eval_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=20,
        # use_safetensors=False,
        learning_rate=lr,
        warmup_ratio=0.1,
        report_to="tensorboard",   # disable wandb
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train() 


if __name__ == "__main__":
    finetune()
    # test_bart_config()