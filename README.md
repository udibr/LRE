<a target="_blank" href="http://twitter.com/udibr"><img alt='Twitter followers' src="https://img.shields.io/twitter/follow/udibr.svg?style=social"></a>

This repository contains source code for the experiments in a paper titled [A Semisupervised Approach for Language Identification based on Ladder Networks](http://arxiv.org/pdf/1604.00317v1.pdf)

In 2015 NIST conducted a [LRE i-vector challenge](https://ivectorchallenge.nist.gov/evaluations/2).
The challenge was to identify which language is spoken from a speech sample, given that the language belongs 
to one of 50 given language or is one of out-of-set languages.
The speech samples were already processed into `i-vectors` and duration information.
The data was split into `training`, `dev` and `test`.
The `training` data included labeled samples from the 50 given languages.
The `dev` data included unlabeled samples from both the 50 given languages and the out-of-set languages.
The `test` was similar to `dev` but it could have been only used for making submissions to the competition.

* [our solution](./A%20Semisupervised%20Approach%20for%20Language%20Identification%20based%20on%20Ladder%20Networks.ipynb) used a modification of the [Ladder Network](http://arxiv.org/abs/1507.02672) and [published code](https://github.com/CuriousAI/ladder).
* [The dark knowledge of tongues](./The%20dark%20knowledge%20of%20tongues.ipynb), fun with the i-vector dataset supplied by the challenge.
