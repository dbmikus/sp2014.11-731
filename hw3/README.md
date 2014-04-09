There are three Python programs here (`-h` for usage):

 - `./decode` a simple non-reordering (monotone) phrase-based decoder
 - `./grade` computes the model score of your output

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./decode | ./grade


The `data/` directory contains the input set to be decoded and the models

 - `data/input` is the input text

 - `data/lm` is the ARPA-format 3-gram language model

 - `data/tm` is the phrase translation model



Experiment and Code Writeup
============================

I implemented a full phrase-based decoder to permit arbitrary reordering during
decoding. To do so, I referenced the following papers:

 - http://acl.ldc.upenn.edu/N/N03/N03-1017.pdf
 - http://www.cs.columbia.edu/~mcollins/pb.pdf

We allow the choice of an arbitrary phrase with the restriction that the end of
the last phrase and the start of the next phrase can at most be distance `d`
from one another. Additionally, we ensure that the earliest untranslated phrase
start is under the distance `d`. This makes sure that we do not end up in a
tight spot where we cannot backtrack to translate that phrase.

When calculating the probability score of a translation, we use the TM logprob
score, the LM logprob score, and a weighting based on the distance between the
previous source phrase and the current source phrase.
We compute a future cost when translating so that we do not always put the
easiest phrases to translate at the start of the sentence. This is calculated
the same way that we calculate the standard probability score except that we
ignore any distance or reordering cost.
We use dynamic programming to come up with a future cost table at the start of
sentence decoding. This way we just look up the future cost from the table
instead of recomputing it for every hypothesis.

I also increased the beam size to 25.
