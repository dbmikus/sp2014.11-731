from collections import defaultdict

# TODO should we include the null string when we do this?
def prepare_iters(bitext):
    # The expected number of times that a given word, f_x translates to e_y
    # throughout the whole corpus
    f_e_expect = defaultdict(float)
    f_source_expect = defaultdict(float)

    for (f, e) in bitext:
        e_count = defaultdict(int)
        f_count = defaultdict(int)
        for e_i in e:
            e_count[e_i] += 1
        for f_j in f:
            f_count[f_j] += 1

        for (f_j, cf) in f_count.items():
            # Determining the number of times that f_j is the source of any
            # translation
            # TODO if we include the null string, remove the + 1 from
            # the denominator
            f_e_expect[f_j] += (float(f_count[f_j]) / (len(f) + 1)) * len(e)

            # Determining the expected number of times f_j translates to e_i in
            # sentence pair (f, e)
            for (e_i, ce) in e_count.items():
                f_source_expect[(f_j, e_i)] += (float(f_count[f_j])
                                                / (len(f) + 1)) * e_count[e_i]

    return (f_e_expect, f_source_expect)


        for (f_j, c) in f_count.items():


def ibm_model1(bitext, f_count, e_count, fe_count):
    # p(e | f, m): this is a dictionary that stores probabilities for
    # translations of (e_i,f_j) pairs
    p_e_fm = defaultdict(int)

    # p(e_i, f_j): translation parameters.
    # We use an EM algorithm to develop this.
    # We have random variables:
    #   f_e = number of times source word f aligns to target word e
    #   a_f = number of times that f was used as the translation source word
    # We start out with uniform probabilities for each translation (f_j, e_i).
    p_e_f = {}
    # p_e_f = defaultdict(lambda k: 1.0 / float(len(fe_count.keys())))

    for (f, e) in bitext:


# For each i in [1, m] (that is, for each empty word spot in the target
# translation), we pick out a word, f_{a_i}, from our source sentence that will
# map to that target spot.
# Note that we do not yet know what word in the target language the word
# f_{a_i} will translate to.
#
# me_iters is the number of ME iterations we do to improve our probabilities.
def prob_pair(p_e_fm, p_e_f, f, e):
    # add the null token as an empty string
    f.insert(0, '')

    # n = length of source sentence
    n = len(f)
    # m = length of target sentence
    m = len(e)

    # Pick out an alignment. For the i'th source word, this
    # picks out a position in the target sentence to translate to.
    a_i_prob = 1.0 / float(n)

    # for each i \in [1,2, ..., m]
    for (i, e_i) in enumerate(e):
        for (j, f_j) in enumerate(f):
            # p(a_i = j): alignment probability
            p_ai_j = a_i_prob

            # Get the expected number of times that f translates into e
            # throughout the whole corpus.
            # This is p(e | f) * number of possible (e, f) pairs in sentences
            if (f_j, e_i) not in p_e_f:


                f_to_e_prob = 1.0 / fe_count[(f_j, e_i)]
                # expected number of times f translates into e throughout the
                # whole corpus
                f_to_e_expect = f_to_e_prob * fe_count[(f_j, e_i)]

                # expected number of times that f is used as the source of any
                # translation
