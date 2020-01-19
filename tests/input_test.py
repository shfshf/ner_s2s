from tokenizer_tools.tagset.offset.sequence import Sequence
from ner_s2s.input import generator_func
from tokenizer_tools.tagset.offset.span import Span


def test_generator_func():
    def data_generator_func():
        seq = Sequence("王小明在北京的清华大学读书。")
        seq.span_set.append(Span(0, 3, 'PERSON'))
        seq.span_set.append(Span(4, 6, 'GPE'))
        seq.span_set.append(Span(7, 11, 'ORG'))

        return [seq]

    config = {
        'preprocess_hook': [{
            'class':
            'seq2annotation.preprocess_hooks.corpus_augment.CorpusAugment'
        }]
    }

    result = generator_func(data_generator_func, config)

    result = [i for i in result]

    assert len(result) == 7
