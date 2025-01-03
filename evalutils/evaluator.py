# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import logging
import sys
from bleu import _bleu, compute_bleu

def evaluate_by_lists(refs, cands):
    assert len(refs) == len(cands)
    
    bleus = []
    for order in [1, 2, 3, 4]:
        #bleu_score, _, _, _, _, _ = compute_bleu([[ref] for ref in refs], cands, order, True)
        bleu_score, _, _, _ = compute_bleu([[ref] for ref in refs], cands, order, True)
        bleu_score = round(bleu_score,3)
        bleus.append({
            f'BLEU-{order}': bleu_score
        })

    result = {
        'BLEUs': bleus,
    }

    return result


def evaluate(ref_path, pred_path):
    refs = [x.strip() for x in open(ref_path, 'r', encoding='utf-8').readlines()]
    pres = [x.strip() for x in open(pred_path, 'r', encoding='utf-8').readlines()]
    
    assert len(refs) == len(pres)

    length = len(refs)
    count = 0
    for i in range(length):
        r = refs[i]
        p = pres[i]
        if r == p:
            count += 1
    
    acc = round(count/length*100, 2)
    
    bleus = []
    for order in [1, 2, 3, 4]:
        bleu_score = round(_bleu(ref_path, pred_path, max_order=order),2)
        bleus.append({
            f'BLEU-{order}': bleu_score
        })

    result = {
        'BLEUs': bleus,
        'Acc': acc 
    }

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate predictions for songshi dataset.')
    parser.add_argument('--references', '-ref',help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-pre',help="filename of the predictions, in txt format.")
    
    args = parser.parse_args()
    result = evaluate(args.references, args.predictions)
    
    print(result)
    
if __name__ == '__main__':
    main()
