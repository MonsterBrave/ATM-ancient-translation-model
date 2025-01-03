import sys
from transformers import BartForConditionalGeneration,BartTokenizer
from datautils import get_tokenized_dataset,get_unidirectional_tokenized_dataset
from config import get_default_args
import torch
import os 
from tqdm import tqdm
sys.path.append("/sda/sun/ATM/evalutils")
from evaluator import evaluate_by_lists


def transfer_datasets(dataset):
    input_ids=[]
    attention_mask=[]
    for batch in dataset:
        input_ids.append(batch['input_ids'])
        attention_mask.append(batch['attention_mask'])
    return {
        "input_ids":torch.tensor(input_ids),
        "attention_mask":torch.tensor(attention_mask)
    }

def inference(args):
    device = torch.device('cuda:1')
    model = BartForConditionalGeneration.from_pretrained(args.ft_model_path)
    # model.to(device)
    tokenizer = BartTokenizer.from_pretrained(args.tokenizer_path,clean_up_tokenization_spaces=True)

    # test_dataset = get_unidirectional_tokenized_dataset(args,tokenizer,split="test")      # use to inference unidirectional_model and bidirectional_model of src+tgt -> tgt+src 
    test_dataset = get_tokenized_dataset(args,tokenizer,split="test")                       # use to inference AMRmask_model: 'AMR/output/finetune/seed_102_m2a or a2m/checkpoint'
    inputs =transfer_datasets(test_dataset)

    print(tokenizer.decode(inputs['input_ids'][0]))
   
    references = []
    for batch in test_dataset:
        references_sentence = tokenizer.decode(batch['labels'],skip_special_tokens=True)
        references.append(references_sentence)

    print(references[0])

    predictions = []
    size = args.batch_size
    # with torch.no_grad():
    for i in tqdm(range(0,len(inputs['input_ids']),size),desc='Inference'):
        inp = torch.tensor(inputs['input_ids'][i:i+size])
        outputs = model.generate(inp,max_length=256,early_stopping=True)
        for output in outputs:
            prediction_sentence = tokenizer.decode(output,skip_special_tokens=True)
            predictions.append(prediction_sentence)

    print(len(references))
    print(len(predictions))
    bleu_score = evaluate_by_lists(refs=references, cands=predictions)
    bleu_result = [str(rs) if  isinstance(rs,dict) else rs for rs in bleu_score['BLEUs']]

    prediction_dir = args.save_prediction_dir
    os.makedirs(prediction_dir, exist_ok=True)
    with open(os.path.join(prediction_dir, "biamr_ft_MaMm2am_a2m_earlymodern.json"), "w") as f:
        f.write("".join(bleu_result)+"\n")
        for s,p in zip(references,predictions):
            f.write("ref:"+"".join(s)+","+"pre:"+"".join(p)+"\n")

def test_model(args):
    text = "帝亟召纲，纲入见，泣拜请死。"
    modern = "皇上立即召见李纲，李纲入宫见皇上，跪拜于地，泪流不止，请求赐死。"

    model = BartForConditionalGeneration.from_pretrained(args.ft_model_path)
    tokenizer = BartTokenizer.from_pretrained(args.tokenizer_path,clean_up_tokenization_spaces=True)
    input = tokenizer(modern,return_tensors='pt')
    decoded_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input['input_ids']]
    print(decoded_texts)
    ancient_id = torch.tensor(tokenizer.convert_tokens_to_ids("<a>"))
    # ip = ancient_id+input['input_ids']
    # print(ip)
    for i in tqdm(range(0,10,2),desc='Inference'):
        print(i)
    output = model.generate(input['input_ids'],max_length=256,num_beams=5,early_stopping=True)
    print(tokenizer.decode(output[0],skip_special_tokens=True))

if __name__=="__main__":
    args = get_default_args()
    inference(args)
    # test_model(args)

