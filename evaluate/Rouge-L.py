import evaluate
import os

def get_rougel(ref_path, gen_path):
    gen_sentence_lst = open(gen_path, encoding='UTF-8').read().split("\n")
    ref_sentence_lst = open(ref_path, encoding='UTF-8').read().split("\n")

    bleu = evaluate.load("rouge")
    results = bleu.compute(predictions=gen_sentence_lst, references=ref_sentence_lst)
    return results['rougeL'] * 100

if __name__ == '__main__':
    ref_path = "../data/angular_filtered/subsets/generation/test_ref.txt"
    gen_path = "../data/angular_filtered/subsets/generation/test_gpt35_golden_classified_rag.txt"

    if os.path.exists(ref_path) and os.path.exists(gen_path):
        print(get_rougel(ref_path, gen_path))
    else:
        print("File not exits")

