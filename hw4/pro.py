"""
Pairwise ranking optimization.
"""

import sys
import itertools, operator, math
import random
from sklearn import linear_model
from sklearn import svm
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
def sampler(meteor_scores, hyps, ref, num_sample, num_ret):
    # Add the additional features here. There is a bunch of pointer and
    # reference stuff going on behind the scenes, so when we have pairs, two
    # separate pairs can share a hypothesis and if we update the features
    # anywhere else, we will update them twice.
    for hyp in hyps:
        hyp[2] = add_feats(hyp[2], hyp[1])
    # Get all of the possible pairs between hypotheses, removing the ones that
    # are paired with themselves.
    all_pairs = list(itertools.combinations(hyps, 2))
    sys.stderr.write("Done generating combinations.\n")
    # Select num_sample random pairs and put them in a list.
    limited_pairs = random.sample(all_pairs, min(num_sample, len(all_pairs)))
    sys.stderr.write("Done getting pairs from combinations.\n")
    # Compare the gold standard scores for the pairs
    pair_scores = []
    for pair in limited_pairs:
        h1_score = meteor_scores[pair[0][1]]
        h2_score = meteor_scores[pair[1][1]]
        if should_add(h1_score - h2_score):
            # We update the feature vectors for the two hypotheses to include
            # features not read in, but instead computed during runtime.
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
        h1_vec = pair_score[0][2]
        h2_vec = pair_score[1][2]
        h1_score = meteor_scores[pair_score[0][1]]
        h2_score = meteor_scores[pair_score[1][1]]
        gold_diff_label = math.copysign(1, h1_score - h2_score)
        # When computing the difference between feature vectors in a pair, we
        # are computing a vector from the second pair element to the first pair
        # element.
        observed_vectors.append(vector_func_combine(operator.sub,
                                                    h1_vec,
                                                    h2_vec))
        targets.append(gold_diff_label)
        # Add the pair in the opposite direction.
        # When computing the difference between feature vectors in a pair, we
        # are computing a vector from the first pair element to the second pair
        # element.
        observed_vectors.append(vector_func_combine(operator.sub,
                                                    h2_vec,
                                                    h1_vec))
        targets.append(-1 * gold_diff_label)
    sys.stderr.write("Done creating vector points and labels.\n")
    return observed_vectors, targets


def vector_func_combine(oper, v1, v2):
    return [oper(vpair[0], vpair[1]) for vpair in zip(v1, v2)]


# We randomly choose n elements in l and put them at the start of l.
def limit_shuffle(l, n):
    l = list(l)
    for i in xrange(min(n, len(l))):
        swap = random.randrange(i, len(l))
        x = l[i]
        l[i] = l[swap]
        l[swap] = x
    return l


def train_classifier(observed_vectors, targets):
    # classifier = linear_model.Ridge(alpha=0.5)
    # classifier = svm.LinearSVC()
    # classifier = linear_model.LogisticRegression()
    classifier = linear_model.SGDRegressor(alpha=1.0)
    classifier.fit(observed_vectors, targets)
    return classifier


# The order of these must match the order of the values we add from add_feats()
def extra_feat_names():
    return ['num_target_words',
            'num_untrans']

# The order of these must match the order of the extra_feat_names()
def add_feats(old_vec, hypothesis):
    concat_vec = []
    # Computing the number of words in the target
    concat_vec.append(len(hypothesis.split()))
    # Computing the number of untranslated words
    concat_vec.append(sum([int(not is_ascii(word))
                           for word
                           in hypothesis.split()]))
    return old_vec + concat_vec

def is_ascii(word):
    try:
        word.decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def sample_and_train_classifier(hyp_train_file, train_ref_file, meteor_scores_file):
    all_hyps = [pair.split(' ||| ') for pair in open(hyp_train_file)]
    all_refs = [ref for ref in open(train_ref_file)]
    all_meteor = [float(score) for score in open(meteor_scores_file)]
    meteor_dict = {}
    for sent_score in zip(all_hyps, all_meteor):
        meteor_dict[sent_score[0][1]] = sent_score[1]
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
            if not set_feat_names:
                feat_names += extra_feat_names()
                set_feat_names = True
        ref = all_refs[s]
        sys.stderr.write("Sampling from sentence %d...\n" % s)
        more_obs_vecs, more_tgts = sampler(meteor_dict, hyps_for_one_sent, ref,
                                           5000, 50)
        sys.stderr.write("Done sampling from sentence %d\n\n" % s)
        observed_vectors += more_obs_vecs
        targets += more_tgts
    trained_clfr = train_classifier(observed_vectors, targets)
    weight_vec = classifier_weight(trained_clfr, feat_names)
    # print weight_vec
    return trained_clfr

def classifier_weight(classifier, feat_names):
    coef = classifier.coef_
    weight_vec = {}
    for i, feat_name in enumerate(feat_names):
        weight_vec[feat_name] = coef[i]
    return weight_vec

if __name__ == '__main__':
    sample_and_train_classifier('data/dev.100best',
                                'data/dev.ref',
                                'meteor-scores.scores')
