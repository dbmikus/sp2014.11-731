There are three Python programs here (`-h` for usage):

 - `./align` aligns words using Dice's coefficient.
 - `./check` checks for out-of-bounds alignment points.
 - `./grade` computes alignment error rate.

The commands are designed to work in a pipeline. For instance, this is a valid
invocation:

    ./align -t 0.9 -n 1000 | ./check | ./grade -n 5


The `data/` directory contains a fragment of the German/English Europarl corpus.

 - `data/dev-test-train.de-en` is the German/English parallel data to be
   aligned. The first 150 sentences are for development; the next 150 is a blind
   set you will be evaluated on; and the remainder of the file is unannotated
   parallel data.

 - `data/dev.align` contains 150 manual alignments corresponding to the first
   150 sentences of the parallel corpus. When you run `./check` these are used
   to compute the alignment error rate. You may use these in any way you choose.
   The notation `i-j` means the word at position *i* (0-indexed) in the German
   sentence is aligned to the word at position *j* in the English sentence; the
   notation `i?j` means they are "probably" aligned.


Algorithms writeup
==================
We perform some normalization on the corpus. I set everything to lowercase.
For each individual digit, I replace it with a "#" sign.
As an example: 1234 --> ####
We also truncate each word down to its first 7 characters.
These normalization techniques managed to bring down our AER a little bit.

I only correctly implemented the IBM Model 1 algorithm. I allow for a variable
number of EM iterations run on top of an initialization step. Although, I
believe that I have a minor bug in this implementation as my algorithm does
not pass the baseline (I am off by around 0.02 AER).

I tried to incorporate the algorithm described here:
http://www.cs.rochester.edu/~gildea/pubs/riley-gildea-acl12.pdf
I ran into overflow issues with the "f" function described for how it applies to
the M step of the EM iterations. I tried just removing the "exp" part of the
function, but this made my AER worse.
