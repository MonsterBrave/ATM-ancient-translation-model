# from peft import LoraConfig, TaskType, get_peft_model

import sys 
from datautils import get_biMerged_tokenized_dataset            # original dataset merged reversed dataset
from datautils import get_AplusM_to_MplusA_tokenized_dataset    # handle dataset to be modern+ancient to ancient+modern
from datautils import get_MaMm2am,get_MaMm2ma

from config import get_default_args
from logging_utils import get_logger

from transformers import(
    BartTokenizer,
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
    BartConfig
)

logger = get_logger(__name__, "info")

def train():
    args = get_default_args()
    logger.info(args)
    set_seed(args.seed)

    tokenizer = BartTokenizer.from_pretrained(args.tokenizer_path, clean_up_tokenization_spaces=True)
    config = BartConfig(
        vocab_size= tokenizer.vocab_size,
        d_model=768,
        encoder_attention_heads=12,
        decoder_attention_heads=12,
        encoder_layers=6,
        decoder_layers=6,
        decoder_ffn_dim=3072,
        encoder_ffn_dim=3072,
        max_position_embeddings=512
    )
    model = BartForConditionalGeneration(config)

    train_dataset = get_MaMm2am(args, tokenizer, 'train',mlm_prob=args.mlm_prob)
    train_len = len(train_dataset)
    eval_dataset = get_MaMm2am(args, tokenizer, 'validation',mlm_prob=args.mlm_prob)

    collator = DataCollatorForSeq2Seq(tokenizer, model)

    # debug
    batch = collator([i for i in train_dataset.select(range(5))])
    logger.debug(batch)

    # lr = 3e-4 if args.peft else 5e-5
    lr = 5e-5
    eval_steps = int(train_len / args.batch_size * 0.2)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=15,
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

def test_bart_config():
    default_config = BartConfig.from_pretrained("facebook/bart-base")  
    print(default_config)   


if __name__ == "__main__":
    train()
    # test_bart_config()