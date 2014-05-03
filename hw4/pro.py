"""
Pairwise ranking optimization.
"""

import sys
import itertools, operator, math
from random import randrange
from sklearn import linear_model
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
    all_pairs = itertools.combinations(hyps, 2)
    sys.stderr.write("Done generating combinations.\n")
    # Select num_sample random pairs and put them in a list.
    limited_pairs = (limit_shuffle(all_pairs, num_sample))[:num_sample]
    sys.stderr.write("Done getting pairs from combinations.\n")
    # Compare the gold standard scores for the pairs
    pair_scores = []
    for pair in limited_pairs:
        h1_score = single_bleu(pair[0][1], ref)
        h2_score = single_bleu(pair[1][1], ref)
        if should_add(h1_score - h2_score):
            pair_scores.append((pair[0], pair[1], abs(h1_score - h2_score)))
    sys.stderr.write("Done making list of golden score differences.\n")
    # Sort the list in descending order by the difference in the gold
    # standard score.
    pair_scores.sort(key=lambda x: -x[2])
    sys.stderr.write("Done sorting list by golden score.\n")
    observed_vectors = []
    targets = []
    # Take the top num_ret pair scores, preferring ones with the greatest gold
    # standard difference. For each pair score, compute the difference between
    # the feature vectors for the hypotheses in the pairs, and record whether
    # the first pair element had a greater gold score than the second pair
    # element.
    for i in xrange(min(num_ret, len(pair_scores))):
        pair_score = pair_scores[i]
        h1_score = single_bleu(pair_score[0][1], ref)
        h2_score = single_bleu(pair_score[1][1], ref)
        gold_diff =  h1_score - h2_score
        gold_diff_sign = math.copysign(1, gold_diff)
        # When computing the difference between feature vectors in a pair, we
        # are computing a vector from the second pair element to the first pair
        # element.
        observed_vectors.append(vector_func_combine(operator.sub,
                                                    pair_score[0][2],
                                                    pair_score[1][2]))
        targets.append(gold_diff_sign)
        # Add the pair in the opposite direction.
        # When computing the difference between feature vectors in a pair, we
        # are computing a vector from the first pair element to the second pair
        # element.
        observed_vectors.append(vector_func_combine(operator.sub,
                                                    pair_score[1][2],
                                                    pair_score[0][2]))
        targets.append(-1.0 * gold_diff_sign)
    sys.stderr.write("Done creating vector points and labels.\n")
    return observed_vectors, targets


def vector_func_combine(oper, v1, v2):
    return [oper(vpair[0], vpair[1]) for vpair in zip(v1, v2)]


# We randomly choose n elements in l and put them at the start of l.
def limit_shuffle(l, n):
    l = list(l)
    for i in xrange(min(n, len(l))):
        swap = randrange(i, len(l))
        x = l[i]
        l[i] = l[swap]
        l[swap] = x
    return l


def train_classifier(observed_vectors, targets):
    classifier = linear_model.Ridge(alpha=0.5)
    classifier.fit(observed_vectors, targets)
    return classifier.coef_

    # clf = linear_model.LogisticRegression()
    # clf.fit(data[0],data[1])
    # return clf.coef_

    # #clf=linear_model.SGDRegressor(alpha='1.0')
    # clf=linear_model.SGDRegressor()
    # clf.fit(training_features,labels)
    # return clf.predict(test_features)


def main():
    all_hyps = [pair.split(' ||| ') for pair in open('data/dev.100best')]
    all_refs = [ref for ref in open('data/dev.ref')]
    observed_vectors = []
    targets = []
    num_sents = len(all_hyps) / 100
    set_feat_names = False
    feat_names = []
    sys.stderr.write("Number of sentences = %d\n" % num_sents)
    for s in xrange(0, num_sents):
        hyps_for_one_sent = all_hyps[s * 100:s * 100 + 100]
        for i, hyp in enumerate(hyps_for_one_sent):
            feats = hyp[2]
            split_feats = []
            for feat in feats.split(' '):
                (k, v) = feat.split('=')
                if not set_feat_names:
                    feat_names.append(k)
                split_feats.append(float(v))
                hyps_for_one_sent[i][2] = split_feats
            set_feat_names = True
        ref = all_refs[s]
        sys.stderr.write("Sampling from sentence %d...\n" % s)
        more_obs_vecs, more_tgts = sampler(hyps_for_one_sent, ref, 5000, 50)
        sys.stderr.write("Done sampling from sentence %d\n\n" % s)
        observed_vectors += more_obs_vecs
        targets += more_tgts
    opt_weight_vec = train_classifier(observed_vectors, targets)
    print opt_weight_vec


if __name__ == '__main__':
    main()
