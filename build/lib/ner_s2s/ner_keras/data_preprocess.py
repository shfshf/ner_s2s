import os
import tensorflow as tf
from tokenizer_tools.tagset.converter.offset_to_biluo import offset_to_biluo
from ioflow.configure import read_configure
from ioflow.corpus import get_corpus_processor
from ner_kashgari.input import generate_tagset, Lookuper, index_table_from_file



config = read_configure()   # ioflow
corpus = get_corpus_processor(config)
corpus.prepare()  # ?
train_data_generator_func = corpus.get_generator_func(corpus.TRAIN)
eval_data_generator_func = corpus.get_generator_func(corpus.EVAL)

corpus_meta_data = corpus.get_meta_info()
tags_data = generate_tagset(corpus_meta_data["tags"])  # process entity into BIO

train_data = list(train_data_generator_func())
eval_data = list(eval_data_generator_func())

tag_lookuper = Lookuper({v: i for i, v in enumerate(tags_data)})  # tag index
vocab_data_file = config.get("vocabulary_file")
vocabulary_lookuper = index_table_from_file(vocab_data_file)    # dict index


def preprocss(data, maxlen):
    raw_x = []
    raw_y = []

    for offset_data in data:
        tags = offset_to_biluo(offset_data)
        words = offset_data.text

        tag_ids = [tag_lookuper.lookup(i) for i in tags]
        word_ids = [vocabulary_lookuper.lookup(i) for i in words]

        raw_x.append(word_ids)
        raw_y.append(tag_ids)

    if maxlen is None:
        maxlen = max(len(s) for s in raw_x)

    print(">>> maxlen: {}".format(maxlen))

    x = tf.keras.preprocessing.sequence.pad_sequences(
        raw_x, maxlen, padding="post"
    )  # right padding

    # lef padded with -1. Indeed, any integer works as it will be masked
    # y_pos = pad_sequences(y_pos, maxlen, value=-1)
    # y_chunk = pad_sequences(y_chunk, maxlen, value=-1)
    y = tf.keras.preprocessing.sequence.pad_sequences(
        raw_y, maxlen, value=0, padding="post"
    )

    return x, y


MAX_SENTENCE_LEN = config.get("max_sentence_len", 25)

train_x, train_y = preprocss(train_data, MAX_SENTENCE_LEN)
test_x, test_y = preprocss(eval_data, MAX_SENTENCE_LEN)