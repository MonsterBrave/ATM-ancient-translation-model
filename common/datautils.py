"""
    dataset use 2 sepcial tokens [A] represented "ancient" and [M] represented "modern",
    random put [A] and [M] in postion [SRC] and [TGT] every batch
"""
from config import get_default_args
from datasets import Dataset,concatenate_datasets
import random
from transformers import AutoTokenizer,BertTokenizer,BartTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence

def get_dataset(args, split='train'):
    file_path = f'{args.data_root}/{args.data_name}/{split}.json'
    dataset = Dataset.from_json(file_path)
    return dataset 

# # no use
# def sentence_mask(tokenizer,inp,mlm_prob=0.35):
#     mask_ids = torch.tensor(tokenizer.mask_token_id)
#     len = inp.shape
#     rand = torch.rand(len)
#     mask_locs = rand > mlm_prob
#     return torch.where(mask_locs,inp,mask_ids)

# handle one sentence
# return a tensor
def joint_mask_sentence(tokens, tokenizer,mlm_prob):
    masked_tokens = [] 
    for token in tokens: 
        if random.random() < mlm_prob: 
            masked_tokens.append(tokenizer.mask_token) 
        else: masked_tokens.append(token) 
    return masked_tokens
   

def get_unidirectional_tokenized_dataset(args, tokenizer, split='train'):
    dataset = get_dataset(args, split)
    def encode_fn(batch):
        src_text = batch[args.src_type]
        tgt_type = 'modern' if args.src_type=='ancient' else 'ancient'
        tgt_text = batch[tgt_type]

        src_encode = tokenizer(src_text,max_length = args.max_length,padding= 'max_length',truncation=True)
        tgt_encode = tokenizer(tgt_text,max_length = args.max_length,padding= 'max_length',truncation=True)

        return {
            'input_ids': src_encode['input_ids'],
            'attention_mask': src_encode['attention_mask'],
            'labels': tgt_encode['input_ids']
        }
    
    fields = ['modern', 'ancient']
    dataset = dataset.map(encode_fn).remove_columns(fields)
    return dataset

def get_tokenized_dataset(args, tokenizer, split='train'):
    dataset = get_dataset(args, split)
    def encode_fn(batch):
        src_text = batch[args.src_type]
        tgt_type = 'modern' if args.src_type=='ancient' else 'ancient'
        tgt_text = batch[tgt_type]

        modern_id = tokenizer.convert_tokens_to_ids("<m>")
        ancient_id = tokenizer.convert_tokens_to_ids("<a>")
        modern_eos_id = tokenizer.convert_tokens_to_ids("</m>")
        ancient_eos_id = tokenizer.convert_tokens_to_ids("</a>")

        src_encode = tokenizer(src_text,max_length = args.max_length,padding= False,truncation=True)
        tgt_encode = tokenizer(tgt_text,max_length = args.max_length,padding= False,truncation=True)
      
        src_type_bos_id = [modern_id] if args.src_type == 'modern' else [ancient_id]
        src_type_eos_id = [modern_eos_id] if args.src_type == 'modern' else [ancient_eos_id]
        tgt_type_bos_id = [ancient_id] if tgt_type == 'ancient' else [modern_id]
        tgt_type_eos_id = [ancient_eos_id] if tgt_type == 'ancient' else [modern_eos_id]

        src_encode = src_type_bos_id + src_encode['input_ids'][1:-1] + src_type_eos_id +[tokenizer.pad_token_id]*(args.max_length-len(src_encode['input_ids']))
        tgt_encode = tgt_type_bos_id + tgt_encode['input_ids'][1:-1] + tgt_type_eos_id + [tokenizer.pad_token_id]*(args.max_length-len(tgt_encode['input_ids']))
        atttention_mask = [1] * len(src_encode)

        return {
            'input_ids': src_encode,     #<a><s>a1,a2,a3</s>
            'attention_mask':atttention_mask,
            'labels': tgt_encode        #<m><s>m1,m2,m3</s>
        }
    
    fields = ['modern', 'ancient']
    dataset = dataset.map(encode_fn).remove_columns(fields)
    return dataset

# def get_bidirectional_tokenized_dataset(args, tokenizer, split='train'):
#     dataset = get_dataset(args, split)
#     def encode_fn(batch):
#         src_type = random.choice(['modern','ancient'])
#         tgt_type = 'modern' if src_type == 'ancient' else 'ancient'
#         src_text = batch[src_type]
#         tgt_text = batch[tgt_type]

#         # modern_id = tokenizer.convert_tokens_to_ids("[M]")
#         # ancient_id = tokenizer.convert_tokens_to_ids("[A]")
#         modern_id = tokenizer.convert_tokens_to_ids("<m>")
#         ancient_id = tokenizer.convert_tokens_to_ids("<a>")


#         # src_encode = tokenizer(src_text)
#         # tgt_encode = tokenizer(tgt_text)
#         src_encode = tokenizer(src_text,max_length = args.max_length,padding= 'max_length',truncation=True)
#         tgt_encode = tokenizer(tgt_text,max_length = args.max_length,padding= 'max_length',truncation=True)
#         src_type_id = [modern_id] if src_type == 'modern' else [ancient_id]
#         tgt_type_id = [ancient_id] if tgt_type == 'ancient' else [modern_id]

#         return {
#             'input_ids': src_type_id + src_encode['input_ids'],
#             'attention_mask': [1] + src_encode['attention_mask'],
#             'labels': tgt_type_id + tgt_encode['input_ids']
#         }
    
#     fields = ['modern', 'ancient']
#     dataset = dataset.map(encode_fn).remove_columns(fields)
#     return dataset

# merge original reverse-dataset 
def get_biMerged_tokenized_dataset(args, tokenizer, split='train'):
    dataset = get_dataset(args, split)
    def encode_reverse_fn(batch):
        src_text = batch['modern']
        tgt_text = batch['ancient']
        modern_bos_id = tokenizer.convert_tokens_to_ids("<m>")
        ancient_bos_id = tokenizer.convert_tokens_to_ids("<a>")
        modern_eos_id = tokenizer.convert_tokens_to_ids("</m>")
        ancient_eos_id = tokenizer.convert_tokens_to_ids("</a>")
        src_encode = tokenizer(src_text,max_length = args.max_length,padding= False,truncation=True)
        tgt_encode = tokenizer(tgt_text,max_length = args.max_length,padding= False,truncation=True)
        src_encode = [modern_bos_id] + src_encode['input_ids'][1:-1] + [modern_eos_id] +[tokenizer.pad_token_id]*(args.max_length-len(src_encode['input_ids']))
        tgt_encode = [ancient_bos_id] + tgt_encode['input_ids'][1:-1] + [ancient_eos_id] + [tokenizer.pad_token_id]*(args.max_length-len(tgt_encode['input_ids']))
        atttention_mask = [1] * len(src_encode)
        return {
            'input_ids': src_encode,
            'attention_mask': atttention_mask,
            'labels': tgt_encode
        }
    
    def encode_fn(batch):
        src_text = batch['ancient']
        tgt_text = batch['modern']
        modern_bos_id = tokenizer.convert_tokens_to_ids("<m>")
        ancient_bos_id = tokenizer.convert_tokens_to_ids("<a>")
        modern_eos_id = tokenizer.convert_tokens_to_ids("</m>")
        ancient_eos_id = tokenizer.convert_tokens_to_ids("</a>")
        src_encode = tokenizer(src_text,max_length = args.max_length,padding= False,truncation=True)
        tgt_encode = tokenizer(tgt_text,max_length = args.max_length,padding= False,truncation=True)
        src_encode = [modern_bos_id] + src_encode['input_ids'][1:-1] + [modern_eos_id] +[tokenizer.pad_token_id]*(args.max_length-len(src_encode['input_ids']))
        tgt_encode = [ancient_bos_id] + tgt_encode['input_ids'][1:-1] + [ancient_eos_id] + [tokenizer.pad_token_id]*(args.max_length-len(tgt_encode['input_ids']))
        atttention_mask = [1] * len(src_encode)
        return {
            'input_ids': src_encode,
            'attention_mask': atttention_mask,
            'labels': tgt_encode
        }
    
    fields = ['modern', 'ancient']
    a2m_dataset = dataset.map(encode_fn).remove_columns(fields)
    m2a_dataset = dataset.map(encode_reverse_fn).remove_columns(fields)
    combined_datasets = concatenate_datasets([a2m_dataset,m2a_dataset])
    return combined_datasets


def get_AplusM_to_MplusA_tokenized_dataset(args,tokenizer,split='train'):
    dataset = get_dataset(args, split)
    def encode_fn(batch):
        modern_type = 'modern'
        ancient_type = 'ancient'
        modern_text = batch[modern_type]
        ancient_text = batch[ancient_type]
        
        src_text = modern_text
        tgt_text = ancient_text
        
        src_encode = tokenizer(src_text,max_length = args.max_length,padding= False,truncation=True)
        tgt_encode = tokenizer(tgt_text,max_length = args.max_length,padding= False,truncation=True)

        splust = src_encode['input_ids']+tgt_encode['input_ids']+[tokenizer.pad_token_id] * (args.max_length-len(src_encode['input_ids']+tgt_encode['input_ids']))
        tpluss = tgt_encode['input_ids']+src_encode['input_ids']+[tokenizer.pad_token_id] * (args.max_length-len(src_encode['input_ids']+tgt_encode['input_ids']))
        mask_attention = [1] * len(splust)
        
        return {
            'input_ids': splust,
            'attention_mask':mask_attention,
            'labels': tpluss
        }
    
    fields = ['modern', 'ancient']
    dataset = dataset.map(encode_fn).remove_columns(fields)
    return dataset

def get_PLUS_data(args, tokenizer, split='test'):
    dataset = get_dataset(args,split)
    def encode_fn(batch):
        src_text = batch[args.src_type]
        tgt_type = 'modern' if args.src_type=='ancient' else 'ancient'
        tgt_text = batch[tgt_type]

        src_encode = tokenizer(src_text,max_length = args.max_length-2,padding= False,truncation=True)
        tgt_encode = tokenizer(tgt_text,max_length = args.max_length-2,padding= False,truncation=True)
        if args.src_type == 'modern':
            input = src_encode['input_ids']+[tokenizer.bos_token_id]+[tokenizer.eos_token_id]+[tokenizer.pad_token_id] * (args.max_length-len(src_encode['input_ids']))
            label = tgt_encode['input_ids']+[tokenizer.bos_token_id]+[tokenizer.eos_token_id]+[tokenizer.pad_token_id] * (args.max_length-len(tgt_encode['input_ids']))
        else :
            input = [tokenizer.bos_token_id]+[tokenizer.eos_token_id]+src_encode['input_ids']+[tokenizer.pad_token_id] * (args.max_length-len(src_encode['input_ids']))
            label = [tokenizer.bos_token_id]+[tokenizer.eos_token_id]+tgt_encode['input_ids']+[tokenizer.pad_token_id] * (args.max_length-len(tgt_encode['input_ids']))
        mask_attention  = [1] * len(input)
        
        return {
            'input_ids': input,
            'attention_mask':mask_attention,
            'labels': label
        }
    
    fields = ['modern', 'ancient']
    dataset = dataset.map(encode_fn).remove_columns(fields)
    return dataset
        

# need change args.max_length=256 and model.max_position_embedding=512
def get_MaMm2ma(args, tokenizer, split='train',mlm_prob=0.35):
    dataset = get_dataset(args, split)
    def encode_fn(batch):
        src_text = batch['ancient']
        tgt_text = batch['modern']

        # byte level encode need first mask then encode
        src_tokens = list(src_text)
        tgt_tokens = list(tgt_text)
        src_input = ''.join(joint_mask_sentence(src_tokens,tokenizer,mlm_prob=mlm_prob))
        tgt_input = ''.join(joint_mask_sentence(tgt_tokens,tokenizer,mlm_prob=mlm_prob))

        modern_bos_id = tokenizer.convert_tokens_to_ids("<m>")
        ancient_bos_id = tokenizer.convert_tokens_to_ids("<a>")
        modern_eos_id = tokenizer.convert_tokens_to_ids("</m>")
        ancient_eos_id = tokenizer.convert_tokens_to_ids("</a>")
        
       
        src_encode = tokenizer(src_input,max_length = args.max_length,padding= False,truncation=True)
        tgt_encode = tokenizer(tgt_input,max_length = args.max_length,padding= False,truncation=True)
        src_label = tokenizer(src_text,max_length = args.max_length,padding= False,truncation=True)
        tgt_label = tokenizer(tgt_text,max_length = args.max_length,padding= False,truncation=True)

        src_encode = [ancient_bos_id] + src_encode['input_ids'][1:-1] + [ancient_eos_id] 
        tgt_encode = [modern_bos_id] + tgt_encode['input_ids'][1:-1] + [modern_eos_id]
        src_label =  [ancient_bos_id] + src_label['input_ids'][1:-1] + [ancient_bos_id] 
        tgt_label =  [modern_bos_id] + tgt_label['input_ids'][1:-1] + [modern_eos_id]

        masked_input = src_encode + tgt_encode + [tokenizer.pad_token_id] * (args.max_length-len(src_encode+tgt_encode))

        labels = tgt_label + src_label + [tokenizer.pad_token_id]*(args.max_length-len(src_label+tgt_label))
        
        attention_mask = [1] * len(masked_input)
        return {
            "input_ids": masked_input, 
            "attention_mask": attention_mask, 
            "labels": labels
        }
    
    fields = ['modern', 'ancient']
    dataset = dataset.map(encode_fn).remove_columns(fields)
    return dataset


# need change args.max_length=256 and model.max_position_embedding=512
def get_MaMm2am(args, tokenizer, split='train',mlm_prob=0.35):
    dataset = get_dataset(args, split)
    def encode_fn(batch):
        src_text = batch['ancient']
        tgt_text = batch['modern']

        # byte level encode need first mask then encode
        src_tokens = list(src_text)
        tgt_tokens = list(tgt_text)
        src_input = ''.join(joint_mask_sentence(src_tokens,tokenizer,mlm_prob=mlm_prob))
        tgt_input = ''.join(joint_mask_sentence(tgt_tokens,tokenizer,mlm_prob=mlm_prob))

        modern_bos_id = tokenizer.convert_tokens_to_ids("<m>")
        ancient_bos_id = tokenizer.convert_tokens_to_ids("<a>")
        modern_eos_id = tokenizer.convert_tokens_to_ids("</m>")
        ancient_eos_id = tokenizer.convert_tokens_to_ids("</a>")
        
       
        src_encode = tokenizer(src_input,max_length = args.max_length,padding= False,truncation=True)
        tgt_encode = tokenizer(tgt_input,max_length = args.max_length,padding= False,truncation=True)
        src_label = tokenizer(src_text,max_length = args.max_length,padding= False,truncation=True)
        tgt_label = tokenizer(tgt_text,max_length = args.max_length,padding= False,truncation=True)

        src_encode = [ancient_bos_id] + src_encode['input_ids'][1:-1] + [ancient_eos_id] 
        tgt_encode = [modern_bos_id] + tgt_encode['input_ids'][1:-1] + [modern_eos_id]
        src_label =  [ancient_bos_id] + src_label['input_ids'][1:-1] + [ancient_bos_id] 
        tgt_label =  [modern_bos_id] + tgt_label['input_ids'][1:-1] + [modern_eos_id]

        masked_input = src_encode + tgt_encode + [tokenizer.pad_token_id] * (args.max_length-len(src_encode+tgt_encode))

        labels = src_label + tgt_label + [tokenizer.pad_token_id]*(args.max_length-len(src_label+tgt_label))
        
        attention_mask = [1] * len(masked_input)
        return {
            "input_ids": masked_input, 
            "attention_mask": attention_mask, 
            "labels": labels
        }
    
    fields = ['modern', 'ancient']
    dataset = dataset.map(encode_fn).remove_columns(fields)
    return dataset


# need change args.max_length=256 and model.max_position_embedding=512
def get_PaPm2partial(args, data, tokenizer, mask= 'ancient',inp='modern',mlm_prob=0.35):
    def encode_fn(batch):
        src_text = batch[inp]
        tgt_text = batch[mask]

        # byte level encode need first mask then encode
        # src_tokens = list(src_text)
        tgt_tokens = list(tgt_text)
        # src_input = ''.join(joint_mask_full(src_tokens,tokenizer,mlm_prob=mlm_prob))
        tgt_input = ''.join(joint_mask_sentence(tgt_tokens,tokenizer,mlm_prob=mlm_prob))

        modern_bos_id = tokenizer.convert_tokens_to_ids("<m>")
        ancient_bos_id = tokenizer.convert_tokens_to_ids("<a>")
        modern_eos_id = tokenizer.convert_tokens_to_ids("</m>")
        ancient_eos_id = tokenizer.convert_tokens_to_ids("</a>")

        src_bos_id,src_eos_id = (modern_bos_id,modern_eos_id) if inp=='modern' else (ancient_bos_id,ancient_eos_id)
        tgt_bos_id,tgt_eos_id = (ancient_bos_id,ancient_eos_id) if mask=='ancient' else (modern_bos_id,modern_eos_id)
       
        src_encode = tokenizer(src_text,max_length = args.max_length,padding= False,truncation=True)
        tgt_encode = tokenizer(tgt_input,max_length = args.max_length,padding= False,truncation=True)
        # src_label = tokenizer(src_text,max_length = args.max_length,padding= False,truncation=True)
        tgt_label = tokenizer(tgt_text,max_length = args.max_length,padding= False,truncation=True)

        src_encode = [src_bos_id] + src_encode['input_ids'][1:-1] +  [src_eos_id]
        tgt_encode = [tgt_bos_id] + tgt_encode['input_ids'][1:-1] + [tgt_eos_id]
        tgt_label =  [tgt_bos_id]  + tgt_label['input_ids'][1:-1] + [tgt_eos_id]

        masked_input = src_encode + tgt_encode + [tokenizer.pad_token_id] * (args.max_length-len(src_encode+tgt_encode))

        labels = tgt_label + [tokenizer.pad_token_id]*(args.max_length-len(tgt_label))
        
        attention_mask = [1] * len(masked_input)
        return {
            "input_ids": masked_input, 
            "attention_mask": attention_mask, 
            "labels": labels
        }
    
    fields = ['modern', 'ancient']
    dataset = data.map(encode_fn).remove_columns(fields)
    return dataset



def random_mask():
    rand = torch.rand([5, 10])
    input_ids = torch.randint(0, 100, (5, 10))
    mask_token_id = torch.tensor(-1)
    mask_locs = rand > 0.5
    print(mask_locs)
    print(input_ids)
    print(torch.where(mask_locs, input_ids, mask_token_id))
    
def test_datasets_num():
    splits = ['train', 'validation','earlymodern','test']
    args = get_default_args()
    testset = get_dataset(args, 'validation')
    print('total_num:',len(testset))




if __name__ == "__main__":
    # random_mask()
    # test_datasets_num()

    splits = ['train', 'validation']
    args = get_default_args()
    tokenizer = BartTokenizer.from_pretrained(args.tokenizer_path, clean_up_tokenization_spaces=True)
    testset = get_dataset(args, 'validation')
    encoded_testset = get_AplusM_to_MplusA_tokenized_dataset(args,tokenizer,'validation')
    print(encoded_testset['input_ids'][0])

    # for i in testset.select(range(10)):
    #     print(i)

    print(encoded_testset)
    print(tokenizer.batch_decode(encoded_testset[:2]['input_ids']))
    print(tokenizer.batch_decode(encoded_testset[:2]['labels']))