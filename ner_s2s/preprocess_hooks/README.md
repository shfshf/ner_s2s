## 对数据进行标点符号的预处理

corpus_augment.py:
对每句话根据几个常用的末尾标点符号分别都处理了一遍

corpus_drop.py:
对每句话的末尾的标点符号进行删除操作

## 在 configure 选择里面配置即可
```
config = {
        'preprocess_hook': [{
            'class':
            'seq2annotation.preprocess_hooks.corpus_augment.CorpusAugment'
        }]
    }
```