---
title: "On AutoML Translate and BLEU"
description: "When evaluating a new test set with AutoML Translate, it can only be done through the UI, and not with the API. To avoid incongruences, use always NLTK to evaluate"
toc: true
comments: true
layout: post
categories: ["Vertex AI"]
image: images/googlecloud.png
author: Rafael Sanchez
---

## Summary

When evaluating a new test set with AutoML Translate, it can only be done through the UI, and [not with the API](https://cloud.google.com/translate/automl/docs/evaluate).
Refer to [here] on how BLEU works. Note the calculation m ay defer if using the open-source tool  and the method `nltk.translate.bleu_score_corpus_bleu`, due to the fact that normalization and tokenization may defer.

So, to avoid misunderstanding between the open source NLTK tool and the internal BLEU scaore calculation in AutoML, use always the NLTK to evaluate.

## Code example

```python
import nltk
hypothesis = ['This', 'is', 'cat'] 
reference = ['This', 'is', 'a', 'cat']
references = [reference] # list of references for 1 sentence.
list_of_references = [references] # list of references for all sentences in corpus.
list_of_hypotheses = [hypothesis] # list of hypotheses that corresponds to list of references.
nltk.translate.bleu_score.corpus_bleu(list_of_references, list_of_hypotheses)
# 0.6025286104785453
nltk.translate.bleu_score.sentence_bleu(references, hypothesis)
# 0.6025286104785453
```

## References

`[1]` [Understanding the BLEU Score](https://cloud.google.com/translate/automl/docs/evaluate#bleu)    
