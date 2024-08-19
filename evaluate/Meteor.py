import evaluate
import os
from nltk.translate import meteor_score

def get_meteor(ref_path, gen_path):
    gen_sentence_lst = open(gen_path, encoding='UTF-8').read().split("\n")
    ref_sentence_lst = open(ref_path, encoding='UTF-8').read().split("\n")

    meteor = evaluate.load("meteor")
    predictions = gen_sentence_lst
    references = ref_sentence_lst
    results = meteor.compute(predictions=predictions, references=references)
    return results['meteor'] * 100

if __name__ == '__main__':
    ref_path = "./data/vdo_filtered/generation/test_ref.txt"
    gen_path = "./data/vdo_filtered/generation/test_gpt35_rag.txt"

    if os.path.exists(ref_path) and os.path.exists(gen_path):
        print(get_meteor(ref_path, gen_path))
    else:
        print("File not exits")

