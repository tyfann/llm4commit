import evaluate
import os


def get_bleu_moses(ref_path, gen_path):
    gen_sentence_lst = open(gen_path, encoding='UTF-8').read().split("\n")
    ref_sentence_lst = open(ref_path, encoding='UTF-8').read().split("\n")

    bleu = evaluate.load("bleu")
    # tolowercase
    gen_sentence_lst = [sentence.lower() for sentence in gen_sentence_lst]
    ref_sentence_lst = [sentence.lower() for sentence in ref_sentence_lst]
    predictions = gen_sentence_lst
    references = ref_sentence_lst
    results = bleu.compute(predictions=predictions, references=references, smooth=True)
    return results['bleu'] * 100


if __name__ == '__main__':
    ref_path = "../data/vdo_filtered/generation/test_ref.txt"
    gen_path = "../data/vdo_filtered/generation/test_gpt35_golden_classified_rag.txt"

    if os.path.exists(ref_path) and os.path.exists(gen_path):
        print(round(get_bleu_moses(ref_path, gen_path), 2))
    else:
        print("File not exits")
