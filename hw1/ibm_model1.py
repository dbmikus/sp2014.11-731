from collections import defaultdict

# TODO should we include the null string when we do this?
def prepare_iters(bitext):
    # The expected number of times that a given word, f_x translates to e_y
    # throughout the whole corpus
    f_e_expect = defaultdict(float)
    f_source_expect = defaultdict(float)

    for (f, e) in bitext:
        # we clone the list so we don't have issues with extra null string
        # insertions in the bitext
        f = list(f)
        # We include the null token as an option for a word translation in the
        # source language.
        f.insert(0, '')
        e_count = defaultdict(int)
        f_count = defaultdict(int)
        for e_i in e:
            e_count[e_i] += 1
        for f_j in f:
            f_count[f_j] += 1

        for (f_j, cf) in f_count.items():
             for (e_i, ce) in e_count.items():
                # Determining the expected number of times f_j translates to
                # e_i in sentence pair (f, e)
                trans_prob = (float(f_count[f_j]) / (len(f))) * e_count[e_i]
                f_e_expect[(f_j, e_i)] += trans_prob
                # Determining the number of times that f_j is the source of any
                # translation
                f_source_expect[f_j] += trans_prob


    # computes the probabilities for all p(e_i | f_j)
    return compute_pef_probs(f_e_expect, f_source_expect)


def compute_pef_probs(f_e_expect, f_source_expect):
    """
    Computes all of the probabilities for translations: p(e_i | f_j)
    """
    p_e_f = defaultdict(int)
    for ((f_j, e_i), trans_exp) in f_e_expect.items():
        p_e_f[(f_j, e_i)] = f_e_expect[(f_j, e_i)] /  f_source_expect[f_j]

    return p_e_f


def ibm_model1(bitext, me_iters):
    (p_e_fm, p_e_f) = run_iterations(bitext, me_iters)

    alignments = get_max_alignments(bitext, p_e_fm, p_e_f)

    # Convert the lines to the string format we need:
    # That is, "%i-%i" <--- substituting in source index and target index
    str_alignments = []
    for sentence_alignment in alignments:
        align_strings = ['%i-%i' % (align_pair[0], align_pair[1])
                         for align_pair in sentence_alignment]
        str_alignments.append(' '.join(align_strings))

    # append a blank string so we end with a newline
    str_alignments.append('')
    return '\n'.join(str_alignments)


def get_max_alignments(bitext, p_e_fm, p_e_f):
    """
    Determine the best alignments for sentences.
    """
    alignments = []
    for (f,e) in bitext:
        sentence_alignment = get_max_sentence_alignment(f, e, p_e_fm, p_e_f)
        alignments.append(sentence_alignment)

    return alignments


def get_max_sentence_alignment(f, e, p_e_fm, p_e_f):
    # we clone the list so we don't have issues with extra null string
    # insertions in the bitext
    f = list(f)
    # add the null token as an empty string
    f.insert(0, '')

    alignments = []
    for (i, e_i) in enumerate(e):
        max_j = None
        max_prob = 0
        for (j, f_j) in enumerate(f):
            if p_e_f[(f_j, e_i)] > max_prob:
                max_prob = p_e_f[(f_j, e_i)]
                max_j = j

        # if it is aligned to null string or there were no non-zero probability
        # alignments, then drop that alignment recording
        if (max_j is not None and max_j != 0):
            alignments.append((max_j, i))

    return alignments


def run_iterations(bitext, me_iters):
    """
    Runs the EM iterations
    """
    # p(e | f, m): this is a dictionary that stores probabilities for
    # translations of (e_i,f_j) pairs.
    # This is a the probability for the translation of sentences.
    p_e_fm = defaultdict(int)

    # p(e_i, f_j): translation parameters.
    # This is the probability for the translation of specific words.
    # We use an EM algorithm to develop this.
    # We have random variables:
    #   f_e = number of times source word f aligns to target word e
    #   a_f = number of times that f was used as the translation source word
    # We start out with uniform probabilities for each translation (f_j, e_i).
    p_e_f = prepare_iters(bitext)
    # p_e_f = defaultdict(lambda k: 1.0 / float(len(fe_count.keys())))

    for i in xrange(me_iters):
        # The expected number of times we translate from a given word f_j to a
        # given word e_i.
        f_e_expect = defaultdict(int)
        f_source_expect = defaultdict(int)
        for (f, e) in bitext:
            (f_e_prob, f_e_expct, f_source_expct) = prob_pair(p_e_f, f, e)
            p_e_fm[(" ".join(f), " ".join(e))] = f_e_prob

            # updating the expected values across the whole corpus
            for ((f_j, e_i), trans_exp) in f_e_expct.items():
                f_e_expect[(f_j, e_i)] += trans_exp
            for (f_j, trans_exp) in f_source_expct.items():
                f_source_expect[f_j] += trans_exp

        # Updating the probabilities for translations between words
        p_e_f = compute_pef_probs(f_e_expect, f_source_expect)

    return (p_e_fm, p_e_f)


# For each i in [1, m] (that is, for each empty word spot in the target
# translation), we pick out a word, f_{a_i}, from our source sentence that will
# map to that target spot.
# Note that we do not yet know what word in the target language the word
# f_{a_i} will translate to.
#
# me_iters is the number of ME iterations we do to improve our probabilities.
def prob_pair(p_e_f, f, e):
    # we clone the list so we don't have issues with extra null string
    # insertions in the bitext
    f = list(f)
    # add the null token as an empty string
    f.insert(0, '')

    # n = length of source sentence
    n = len(f)
    # m = length of target sentence
    m = len(e)

    # Pick out an alignment. For the i'th source word, this
    # picks out a position in the target sentence to translate to.
    a_i_prob = 1.0 / float(n)

    # The probability we generate is:
    # p(e | f, m) = \prod_{i=1}^m  \sum_{j=0}^n  p(a_i = j) x p(e_i | f_j)

    prob_prod = 1

    # Expected number of translations from some f_j to some e_i
    f_e_expect = defaultdict(int)
    f_source_expect = defaultdict(int)

    # for each i \in [1,2, ..., m]
    for (i, e_i) in enumerate(e):
        # We must compute the sum of probabilities for translating from all
        # words f_j1, f_j2, etc. in the source sentence f, to word e_i in the
        # target sentence e.
        prob_sum = 0
        for (j, f_j) in enumerate(f):
            # Computing the expected values that we use for our iterative EM.
            # These expected values are used to give us the new p(e_i | f_j)
            # probabilities at the end of each iteration.
            f_e_expect[(f_j, e_i)] += p_e_f[(f_j, e_i)]
            f_source_expect[f_j] += p_e_f[(f_j, e_i)]


            # Computing the probability relevant to the probabalistic expression
            # we care about in the end.
            # That is, for target sentence e and source sentence f and target
            # sentence length m:
            #   p(e | f, m)

            # p(a_i = j): alignment probability
            p_ai_j = a_i_prob

            prob_sum += p_ai_j * p_e_f[(f_j, e_i)]

        prob_prod *= prob_sum

    return (prob_prod, f_e_expect, f_source_expect)
