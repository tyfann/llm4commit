import os, argparse
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


def get_bleu_nltk(ref_path, gen_path):
    gen_sentence_lst = open(gen_path, encoding='UTF-8').read().split("\n")
    ref_sentence_lst = open(ref_path, encoding='UTF-8').read().split("\n")
    sentence_bleu_lst = [
        sentence_bleu([ref_sentence.split()], gen_sentence.split(), smoothing_function=SmoothingFunction().method5) for
        ref_sentence, gen_sentence in zip(ref_sentence_lst, gen_sentence_lst)]
    stc_bleu = np.mean(sentence_bleu_lst)
    return stc_bleu * 100


if __name__ == "__main__":

    ref_path = "../data/angular_filtered/subsets/generation/test_ref.txt"
    gen_path = "../data/angular_filtered/subsets/generation/test_gpt35_golden_classified_rag.txt"

    if os.path.exists(ref_path) and os.path.exists(gen_path):
        print(get_bleu_nltk(ref_path, gen_path))
    else:
        print("File not exits")
