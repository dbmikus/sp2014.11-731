# -*- coding: utf-8 -*-

from collections import defaultdict

verbose = False
def vprint(string):
    """
    Prints the input string if verbose mode is on.
    """
    if verbose:
        print string

# TODO should we include the null string when we do this?
def prepare_iters(bitext):
    # The number of times that a given word, f_x, could translate to e_y
    # throughout the whole corpus
    f_e_count = defaultdict(int)
    # The number of times that a given word, f_x, could translate to
    # anything
    f_source_count = defaultdict(int)

    for (f, e) in bitext:
        # we clone the list so we don't have issues with extra null string
        # insertions in the bitext
        f = list(f)
        # We include the null token as an option for a word translation in the
        # source language.
        f.insert(0, '')
        for (j, f_j) in enumerate(f):
            for (i, e_i) in enumerate(e):
                # We are assuming that p(e_i | f_j) is uniform.
                # Any f_j has an equal chance of translating to any e_i.
                # We just count up the number of times that f_j occurs,
                # and the number of times that f_j can be paired up with e_i in
                # a given sentence.
                # We can then normalize these to get the default uniform
                # probabilities for p(e_i | f_j)
                f_e_count[(f_j, e_i)] += 1
                f_source_count[f_j] += 1

    # Normalize the counts and use this as our starting probabilities
    # for p(e_i | f_j)
    return compute_pef_probs(f_e_count, f_source_count)



def compute_pef_probs(f_e_expect, f_source_expect):
    """
    Computes all of the probabilities for translations: p(e_i | f_j)
    """
    p_e_f = defaultdict(int)
    for ((f_j, e_i), trans_exp) in f_e_expect.items():
        vprint('f_j = ' + f_j)
        vprint('e_i = ' + e_i)
        vprint('trans_exp = ' + str(trans_exp))
        vprint('f_source_expect[f_j] = ' + str(f_source_expect[f_j]))
        vprint('')
        p_e_f[(f_j, e_i)] = float(trans_exp) /  float(f_source_expect[f_j])

    vprint('done computing')
    vprint(p_e_f)
    return p_e_f


def ibm_model1(bitext, me_iters):
    p_e_f = run_iterations(bitext, me_iters)

    alignments = get_max_alignments(bitext, p_e_f)

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


def get_max_alignments(bitext, p_e_f):
    """
    Determine the best alignments for sentences.
    """
    alignments = []
    for (f,e) in bitext:
        sentence_alignment = get_max_sentence_alignment(f, e, p_e_f)
        alignments.append(sentence_alignment)

    return alignments


def get_max_sentence_alignment(f, e, p_e_f):
    # we clone the list so we don't have issues with extra null string
    # insertions in the bitext
    f = list(f)
    # add the null token as an empty string
    f.insert(0, '')

    alignments = []

    # This allows a source word to translate to more than one target word,
    # since for each target word we have the possibility of selecting any
    # source word.
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


# me_iters is the number of ME iterations we do to improve our probabilities.
def run_iterations(bitext, me_iters):
    """
    Runs the EM iterations
    """
    # p(e_i, f_j): translation parameters.
    # This is the probability for the translation of specific words.
    # We use an EM algorithm to develop this.
    # We have random variables:
    #   f_e = number of times source word f aligns to target word e
    #   a_f = number of times that f was used as the translation source word
    # We start out with uniform probabilities for each translation (f_j, e_i).
    p_e_f = prepare_iters(bitext)
    vprint(p_e_f)

    for i in xrange(me_iters):
        vprint("On iteration " + str(i))
        # The expected number of times we translate from a given word f_j to a
        # given word e_i.
        f_e_expect = defaultdict(int)
        f_source_expect = defaultdict(int)
        for (f, e) in bitext:
            (f_e_expct, f_source_expct) = update_pair_counts(p_e_f, f, e)

            # updating the expected values across the whole corpus
            for ((f_j, e_i), trans_exp) in f_e_expct.items():
                f_e_expect[(f_j, e_i)] += trans_exp
            for (f_j, trans_exp) in f_source_expct.items():
                f_source_expect[f_j] += trans_exp

        # Updating the probabilities for translations between words
        p_e_f = compute_pef_probs(f_e_expect, f_source_expect)

    return p_e_f


# We update the expected counts for the number of translations from f_j to e_i
# and the expected number of translations from f_j to any word in the target
# language.
def update_pair_counts(p_e_f, f, e):
    # we clone the list so we don't have issues with extra null string
    # insertions in the bitext
    f = list(f)
    # add the null token as an empty string
    f.insert(0, '')

    # Expected number of translations from some f_j to some e_i
    f_e_expect = defaultdict(int)
    f_source_expect = defaultdict(int)

    for (j, f_j) in enumerate(f):
        # for each i \in [1,2, ..., m]
        for (i, e_i) in enumerate(e):
            vprint('p_e_f[(' + f_j + ',' + e_i + ')] = ' + str(p_e_f[(f_j, e_i)]))
            # Computing the expected values that we use for our iterative EM.
            # These expected values are used to give us the new p(e_i | f_j)
            # probabilities at the end of each iteration.
            f_e_expect[(f_j, e_i)] += p_e_f[(f_j, e_i)]
            f_source_expect[f_j] += p_e_f[(f_j, e_i)]

    return (f_e_expect, f_source_expect)
