import evaluate
import os

def get_bleu_moses(ref_path, gen_path):
    gen_sentence_lst = open(gen_path, encoding='UTF-8').read().split("\n")
    ref_sentence_lst = open(ref_path, encoding='UTF-8').read().split("\n")

    bleu = evaluate.load("bleu")
    predictions = gen_sentence_lst
    references = ref_sentence_lst
    results = bleu.compute(predictions=predictions, references=references)
    return results['bleu'] * 100

if __name__ == '__main__':
    ref_path = "../data/angular_filtered/subsets/generation/test_ref.txt"
    gen_path = "../data/angular_filtered/subsets/generation/test_race_v1.txt"

    if os.path.exists(ref_path) and os.path.exists(gen_path):
        print(get_bleu_moses(ref_path, gen_path))
    else:
        print("File not exits")

