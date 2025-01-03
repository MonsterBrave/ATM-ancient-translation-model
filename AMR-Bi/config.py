import os
import argparse

def get_default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default="/sda/sun/ATM/data", type=str)
    parser.add_argument('--data_name', default="Ming-History", type=str)
    parser.add_argument('--seed', default="101", type=int)
    parser.add_argument('--tokenizer_path',default="/sda/sun/ATM/tokenizers/niutrans-bpe-tokenizer",type=str)
    parser.add_argument('--batch_size', default="8", type=int)
    parser.add_argument('--peft', default='false', type=str)
    parser.add_argument('--max_length',default="256",type=int)
    parser.add_argument('--visable_device', default="0", type=str)
    parser.add_argument('--mlm_prob',default=0.35,type=float)

    parser.add_argument('--save_prediction_dir',default="./output/prediction",type=str) 
    parser.add_argument('--output_dir', default="./output/Ming-History/pretrain/MaMm2am", type=str)  # 首先训AM->am
    parser.add_argument('--ft_output_dir',default="./output/Book-of-Han/finetune/MaMm2am", type=str)

    parser.add_argument('--model_path', default="/sda/sun/ATM/model/AMR-Bi/output/Book-of-Han/pretrain/MaMm2am/seed_101/checkpoint-27030", type=str)
    parser.add_argument('--ft_model_path',default='/sda/sun/ATM/model/AMR-Bi/output/earlyModern/finetune/MaMm2am/seed_101_a2m/checkpoint-26040',type=str)
    parser.add_argument('--src_type',default='ancient', type=str)
    parser.add_argument('--direction',default="a2m", type=str)

    # parser.add_argument('--weight_decay', default=0.0, type=float)    # use in AMR_AR model training
    # parser.add_argument('--num_train_epochs', default=10, type=int)
    # parser.add_argument('--gradient_accumulation_steps', default=2, type=int)
    # parser.add_argument('--learning_rate', default=5e-5, type=float)
    # parser.add_argument('--warmup_ratio', default=0.1, type=float)
    # parser.add_argument('--mlm_ancient_plus_modern', action="store_true")
    # parser.add_argument('--fp16',action="store_true")
    # parser.add_argument('--logging_steps', default=20, type=int)
    # parser.add_argument('--save_total_limit', default=2, type=int)
    # parser.add_argument('--do_train',action="store_true")
    # parser.add_argument('--do_eval',action="store_true")
    # parser.add_argument('--evaluate_during_training',action="store_true")
    # parser.add_argument('--max_steps', default=7000, type=int)
    # parser.add_argument('--no_cuda',action="store_true")
    # parser.add_argument('--fp16_opt_level',default="O1",type=str)
    # parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    args = parser.parse_args()
    
    # args.output_dir = f"{args.output_dir}/seed_{args.seed}/{args.direction}"      # used to train: Uni-Bart(m2a,a2m); AMR-Bi(a+m2m,a+m2a)
    args.output_dir = f"{args.output_dir}/seed_{args.seed}"                         # used to train : SPE-Bi; PLUS-Bi; AMR(a+m2m+a,a+m2a+m)
    
    args.ft_output_dir = f"{args.ft_output_dir}/seed_{args.seed}_{args.direction}"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visable_device

    
    return args