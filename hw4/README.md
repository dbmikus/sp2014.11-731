Algorithm and Experiment Description
----------------------------------------

I implemented a basic pairwise ranking optimization. For each set of source to
100-best list of hypothesis translations, we sample 5,000 random unique pairs.
We then keep the first 50 pairs with the greatest difference in golden score.
For our golden score function, we computed the METEOR score for each hypothesis
related to its reference translation.

I added a few features to our feature vector on top of the default features.
Following is the list of all of the features:
    - Translation model p(e|f) weight
    - Language model p(e) weight
    - Lexical translation model p_lex(f|e) weight
    - Length of target sentence len(tgt)
    - Number of untranslated words in the target sentence


How to Run
--------------------
If `./meteor-scores.scores` does not exist, you must run:
    get_meteor_scores.py

Then, to run and evaluate the scoring:
    ./rerank | ./score-meteor
