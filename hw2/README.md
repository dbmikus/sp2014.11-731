METEOR scoring
========================================
The only successful work that I had was my implementation of the METEOR scoring
system. My meteor scoring system did a few things:

1. Create a stemmed version of the reference and hypothesis sentence.
2. Calculate the intersection set of words between the reference sentence and
   the hypothesis sentence. Do the same for the stemmed version.
3. Calculate the precision and recall for the hypothesis sentence when compared
   to the reference sentence. Do the same for the stemmed version.
4. Compute a weighted harmonic mean of the precision and recall. We combine the
   precision for stemmed and full words into one precision score, and we do the
   same for the recall.
5. Compute a penalty score based on shared chunks. A chunk is a series of
   adjacent words. To compute the number of chunks, we continually find the
   longest common substring and then remove that substring from hypothesis. We
   do this until there are none left. For each one that we remove, we increase
   the number of chunks we've counted. We only remove from the hypothesis
   sequence. This is because each unigram in the hypothesis can map to at most
   one unigram in the reference. Thus, once we have mapped a chunk from the
   hypothesis, that chunk can no longer map. However, a chunk in the reference
   can cover two separate chunks in the hypothesis.
6. We count the number of total shared chunks for both the full and stemmed
   sentences, and we also sum up the number of unigrams from those chunks. This
   becomes a penalty ration.
7. Given our harmonic mean, `f_mean`, our chunk penalty score `chunk_penalty`,
   and parameters `beta` and `gamma`, we adjust our score to be:
       `f_mean * (1 - (gamma * (chunk_penalty**beta)))`
8. We return this final score as our METEOR score.


Failed attempts
========================================
I tried a number of more advanced things, but I could not get any of them to
improve the score, and I ran out of time to iterate on the approaches to
determine if they were actually worthwhile.

Splitting Compounds and Complex words
----------------------------------------
I used the Morfessor library to split compounds. You can see my code by checking
out the `compound-split' branch. I trained this on the europarl Czech
data. Unfortunately, under my observations and techniques, splitting the
compound words and complex words into simpler morphemic units did not improve my
evaluation score.

Word vectors
----------------------------------------
I used the word2vec library to create word vectors with distances between
different words. I wanted to use this to sum up the distance from a given word
in reference to every word in the hypothesis, and then sum up these values for
each reference word. This was far too slow, and I ran out of time to read
research papers and determine if there was a quicker way to accomplish a similar
goal.


Sentence edit distance
----------------------------------------
I calculated string edit distance, but instead of by character it operated by
word. This also did not have a desirable result, so I eschewed working on it.


<!---
Local Variables:
mode: markdown
fill-column: 80
eval: (auto-fill-mode 1)
End:
-->
