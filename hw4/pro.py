"""
Pairwise ranking optimization.
"""

import itertools, operator
from random import randrange
import bleu

def single_bleu(hyp, ref):
    stats = [0 for i in xrange(10)]
    stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(hyp,ref))]
    return bleu.bleu(stats)


# Function to determine if we should add a value to our pair_scores list.
def should_add(x):
    if abs(x) < 0.05:
        return False
    else:
        return True

# Selects num_sample random hypothesis pairs, and then takes the num_ret
# greatest difference pairs and computes the difference between their feature
# vectors, along with which pair element had a greater gold score.
# hyps is of the form:
#   (num, hyp, features)
def sampler(hyps, ref, num_sample, num_ret):
    # Get all of the possible pairs between hypotheses, removing the ones that
    # are paired with themselves.
    all_pairs = list(itertools.combinations(hyps, 2))
    # Select num_sample random pairs and put them in a list.
    limited_pairs = (limit_shuffle(all_pairs, num_sample))[:num_sample]
    # Compare the gold standard scores for the pairs
    pair_scores = []
    for pair in limited_pairs:
        h1_score = single_bleu(pair[0][1], ref)
        h2_score = single_bleu(pair[1][1], ref)
        if should_add(h1_score - h2_score):
            pair_scores.append((pair[0], pair[1], abs(h1_score - h2_score)))
    # Sort the list in descending order by the difference in the gold
    # standard score.
    pair_scores.sort(key=lambda x: -x[2])
    ret_list = []
    # Take the top num_ret pair scores, preferring ones with the greatest gold
    # standard difference. For each pair score, compute the difference between
    # the feature vectors for the hypotheses in the pairs, and record whether
    # the first pair element had a greater gold score than the second pair
    # element.
    for i in xrange(min(num_ret, len(pair_scores))):
        pair_score = pair_scores[i]
        h1_gt_h2 = (single_bleu(pair_score[0][1], ref)
                    > single_bleu(pair_score[1][1], ref))
        # When computing the difference between feature vectors in a pair, we
        # are computing a vector from the second pair element to the first pair
        # element.
        ret_list.append((vector_func_combine(operator.sub,
                                             pair_score[0][2], pair_score[1][2]),
                         h1_gt_h2))
        # Add the pair in the opposite direction.
        # When computing the difference between feature vectors in a pair, we
        # are computing a vector from the first pair element to the second pair
        # element.
        ret_list.append((vector_func_combine(operator.sub,
                                             pair_score[1][2], pair_score[0][2]),
                         not h1_gt_h2))
    return ret_list


def vector_func_combine(oper, v1, v2):
    return [(kv_pair[0][0], oper(kv_pair[0][1], kv_pair[1][1]))
            for kv_pair in zip(v1, v2)]


# We randomly choose n elements in l and put them at the start of l.
def limit_shuffle(l, n):
    l = list(l)
    for i in xrange(min(n, len(l))):
        swap = randrange(i, len(l))
        x = l[i]
        l[i] = l[swap]
        l[swap] = x
    return l


def main():
    all_hyps = [pair.split(' ||| ') for pair in open('data/dev.100best')]
    all_refs = [ref for ref in open('data/dev.ref')]
    num_sents = len(all_hyps) / 100
    for s in xrange(0, num_sents):
        hyps_for_one_sent = all_hyps[s * 100:s * 100 + 100]
        for i, hyp in enumerate(hyps_for_one_sent):
            feats = hyp[2]
            split_feats = []
            for feat in feats.split(' '):
                (k, v) = feat.split('=')
                split_feats.append((k,float(v)))
                hyps_for_one_sent[i][2] = split_feats
        ref = all_refs[s]
        print sampler(hyps_for_one_sent, ref, 5000, 50)[2]
        return

if __name__ == '__main__':
    main()
