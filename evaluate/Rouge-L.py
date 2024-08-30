import evaluate
import os
from rouge_score import rouge_scorer, scoring
from rouge_score import tokenizers


def _lcs_table(ref, can):
    """Create 2-d LCS score table."""
    rows = len(ref)
    cols = len(can)
    lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if ref[i - 1] == can[j - 1]:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
    return lcs_table


def cal_lcs(ref_sentence_lst, gen_sentence_lst):
    tokenizer = tokenizers.DefaultTokenizer()
    target_tokens = tokenizer.tokenize(ref_sentence_lst)
    prediction_tokens = tokenizer.tokenize(gen_sentence_lst)

    lcs_table = _lcs_table(target_tokens, prediction_tokens)
    lcs_length = lcs_table[-1][-1]

    return lcs_length


def cal_avg_lcs(ref_path, gen_path):
    gen_sentence_lst = open(gen_path, encoding='UTF-8').read().split("\n")
    ref_sentence_lst = open(ref_path, encoding='UTF-8').read().split("\n")
    lcs_lst = []
    for ref, gen in zip(ref_sentence_lst, gen_sentence_lst):
        lcs_lst.append(cal_lcs(ref, gen))
    return sum(lcs_lst) / len(lcs_lst)


def get_rougel(ref_path, gen_path):
    gen_sentence_lst = open(gen_path, encoding='UTF-8').read().split("\n")
    ref_sentence_lst = open(ref_path, encoding='UTF-8').read().split("\n")

    bleu = evaluate.load("rouge")
    results = bleu.compute(predictions=gen_sentence_lst, references=ref_sentence_lst)
    return results['rougeL'] * 100


if __name__ == '__main__':
    ref_path = "../data/vdo_filtered/generation/test_ref.txt"
    gen_path = "../data/vdo_filtered/generation/test_race.txt"

    if os.path.exists(ref_path) and os.path.exists(gen_path):
        print(round(get_rougel(ref_path, gen_path), 2))
    else:
        print("File not exits")
