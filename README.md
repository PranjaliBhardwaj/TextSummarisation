# TextSummarisation
Text summarization using machine learning involves leveraging algorithms to automatically generate concise and coherent summaries of longer texts. There are two main approaches to this task:

Extractive Summarization: This method identifies and extracts the most important sentences or phrases from the original text to create a summary. Machine learning models, such as those based on attention mechanisms or clustering algorithms, are used to determine which parts of the text are most relevant.

Abstractive Summarization: Unlike extractive methods, abstractive summarization generates new sentences to convey the main ideas of the text. This approach often involves advanced techniques such as sequence-to-sequence models and transformers (e.g., BERT, GPT) to understand context and rephrase content in a more human-like manner.

Machine learning models trained on large datasets can learn to identify key information and produce summaries that are both informative and concise. These models are continually improving, leading to more accurate and coherent text summarization.

## Large-Scale Multi-Document Summarization Dataset and Abstractive Hierarchical Model
Automatic generation of summaries from multiple news articles is a valuable tool as the
number of online publications grows rapidly.
Single document summarization (SDS) systems have benefited from advances in neural encoder-decoder model thanks to the availability of large datasets. However, multidocument summarization (MDS) of news articles has been limited to datasets of a couple
of hundred examples. In this paper, we introduce Multi-News, the first large-scale MDS
news dataset. Additionally, we propose an
end-to-end model which incorporates a traditional extractive summarization model with a
standard SDS model and achieves competitive
results on MDS datasets. We benchmark several methods on Multi-News and release our
data and code in hope that this work will promote advances in summarization in the multidocument setting.

##  Using Pegasus
Recent work pre-training Transformers with
self-supervised objectives on large text corpora
has shown great success when fine-tuned on
downstream NLP tasks including text summarization. However, pre-training objectives tailored for abstractive text summarization have
not been explored. Furthermore there is a
lack of systematic evaluation across diverse domains. In this work, we propose pre-training
large Transformer-based encoder-decoder models on massive text corpora with a new selfsupervised objective. In PEGASUS, important
sentences are removed/masked from an input document and are generated together as one output
sequence from the remaining sentences, similar
to an extractive summary. We evaluated our best
PEGASUS model on 12 downstream summarization tasks spanning news, science, stories, instructions, emails, patents, and legislative bills. Experiments demonstrate it achieves state-of-the-art performance on all 12 downstream datasets measured
by ROUGE scores. Our model also shows surprising performance on low-resource summarization,
surpassing previous state-of-the-art results on 6
datasets with only 1000 examples. Finally we
validated our results using human evaluation and
show that our model summaries achieve human
performance on multiple datasets.

## Summarization with Pointer-Generator Networks

Neural sequence-to-sequence models have
provided a viable new approach for abstractive text summarization (meaning
they are not restricted to simply selecting
and rearranging passages from the original text). However, these models have two
shortcomings: they are liable to reproduce
factual details inaccurately, and they tend
to repeat themselves. In this work we propose a novel architecture that augments the
standard sequence-to-sequence attentional
model in two orthogonal ways. First,
we use a hybrid pointer-generator network
that can copy words from the source text
via pointing, which aids accurate reproduction of information, while retaining the
ability to produce novel words through the
generator. Second, we use coverage to
keep track of what has been summarized,
which discourages repetition. We apply
our model to the CNN / Daily Mail summarization task, outperforming the current
abstractive state-of-the-art by at least 2
ROUGE points.

## Unified Text-to-Text Transformer
Transfer learning, where a model is first pre-trained on a data-rich task before being finetuned on a downstream task, has emerged as a powerful technique in natural language
processing (NLP). The effectiveness of transfer learning has given rise to a diversity of
approaches, methodology, and practice. In this paper, we explore the landscape of transfer
learning techniques for NLP by introducing a unified framework that converts all text-based
language problems into a text-to-text format. Our systematic study compares pre-training
objectives, architectures, unlabeled data sets, transfer approaches, and other factors on
dozens of language understanding tasks. By combining the insights from our exploration
with scale and our new “Colossal Clean Crawled Corpus”, we achieve state-of-the-art results
on many benchmarks covering summarization, question answering, text classification, and
more. To facilitate future work on transfer learning for NLP, we release our data set,
pre-trained models, and code.1
Keywords: transfer learning, natural language processing, multi-task learning, attentionbased models, deep learning.


## Deep Bidirectional Transformers for Language Understanding

We introduce a new language representation model called BERT, which stands for
Bidirectional Encoder Representations from
Transformers. Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pretrain deep bidirectional representations from
unlabeled text by jointly conditioning on both
left and right context in all layers. As a result, the pre-trained BERT model can be finetuned with just one additional output layer
to create state-of-the-art models for a wide
range of tasks, such as question answering and
language inference, without substantial taskspecific architecture modifications.
BERT is conceptually simple and empirically
powerful. It obtains new state-of-the-art results on eleven natural language processing
tasks, including pushing the GLUE score to
80.5% (7.7% point absolute improvement),
MultiNLI accuracy to 86.7% (4.6% absolute
improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1
(5.1 point absolute improvement).


