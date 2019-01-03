import tensorflow as tf
import random
from tqdm import tqdm
import spacy
import ujson as json
from collections import Counter
import numpy as np
from codecs import open

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''

nlp = spacy.blank("en")


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current) # 字符串查找S.find(sub, start=None, end=None),返回被找到的第一个sub下标
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, data_type, word_counter, char_counter):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                context_tokens = word_tokenize(context) # ['first_word', 'second_word', ... ]
                # [['every char of one_word'], ['every char of one_word'], ... ]
                context_chars = [list(token) for token in context_tokens]
                # 返回context_tokens列表中每个token在context中的起始span [(start, end), (start, end), ...]
                spans = convert_idx(context, context_tokens)
                for token in context_tokens:
                    word_counter[token] += len(para["qas"]) # 某一para下每一个问题都让这个para中的词被统计一次
                    for char in token:
                        char_counter[char] += len(para["qas"]) # 同理与char统计频率
                for qa in para["qas"]:
                    total += 1 # 统计总共有多少个question
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1 # 对question中的词频统计
                        for char in token:
                            char_counter[char] += 1 # 对question中的char频率统计
                    y1s, y2s = [], []
                    answer_texts = [] # 收集一个question的所有answer_text
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            # 当答案结束位置小于等于某个单词的开始位置，显然这个单词不是答案；同理
                            # 当答案开始位置大于等于某个单词的结束位置，也不是答案
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx) # 答案基于整个para的编号
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                    example = {"context_tokens": context_tokens, "context_chars": context_chars, "ques_tokens": ques_tokens,
                               "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "id": total}
                    examples.append(example)
                    eval_examples[str(total)] = { # eval_examples通过examples['id']来唯一索引一个example
                        "context": context, "spans": spans, "answers": answer_texts, "uuid": qa["id"]}
        random.shuffle(examples)
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples


def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    # 将字典里的词频大于limit的词，依次记录在list中
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            # 遍历每一行，找到符合要求的word-vector组合
            for line in tqdm(fh, total=size):
                array = line.split()
                word = "".join(array[0:-vec_size]) # get word
                vector = list(map(float, array[-vec_size:])) # get word vector
                if word in counter and counter[word] > limit: # 对在counter中，并且大于limit的词embeding
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        # 如果不使用GloVe，则对每一个token随机初始化一个vector
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    # 从2开始enumerate，编码embedding_dict中的词，  其中0：--NULL--   1：--OOV--
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    # 根据token2idx_dict，生成idx --> token对应的emb_dict，再将idx转化为emb_mat矩阵的行下标。
    idx2emb_dict = {idx: embedding_dict[token] for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict

def convert_to_features(config, data, word2idx_dict, char2idx_dict):

    example = {}
    context, question = data
    context = context.replace("''", '" ').replace("``", '" ')
    question = question.replace("''", '" ').replace("``", '" ')
    example['context_tokens'] = word_tokenize(context)
    example['ques_tokens'] = word_tokenize(question)
    example['context_chars'] = [list(token) for token in example['context_tokens']]
    example['ques_chars'] = [list(token) for token in example['ques_tokens']]

    para_limit = config.test_para_limit
    ques_limit = config.test_ques_limit
    ans_limit = 100
    char_limit = config.char_limit

    def filter_func(example):
        return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit

    if filter_func(example):
        raise ValueError("Context/Questions lengths are over the limit")

    context_idxs = np.zeros([para_limit], dtype=np.int32)
    context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
    ques_idxs = np.zeros([ques_limit], dtype=np.int32)
    ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
    y1 = np.zeros([para_limit], dtype=np.float32)
    y2 = np.zeros([para_limit], dtype=np.float32)

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1

    for i, token in enumerate(example["context_tokens"]):
        context_idxs[i] = _get_word(token)

    for i, token in enumerate(example["ques_tokens"]):
        ques_idxs[i] = _get_word(token)

    for i, token in enumerate(example["context_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            context_char_idxs[i, j] = _get_char(char)

    for i, token in enumerate(example["ques_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            ques_char_idxs[i, j] = _get_char(char)

    return context_idxs, context_char_idxs, ques_idxs, ques_char_idxs

def build_features(config, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):

    para_limit = config.test_para_limit if is_test else config.para_limit
    ques_limit = config.test_ques_limit if is_test else config.ques_limit
    ans_limit = 100 if is_test else config.ans_limit
    char_limit = config.char_limit

    def filter_func(example, is_test=False):
        return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit or \
               (example["y2s"][0] - example["y1s"][0]) > ans_limit

    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    total_ = 0
    meta = {}
    for example in tqdm(examples):
        total_ += 1

        if filter_func(example, is_test):
            #如果满足条件，这处理；否则跳过，进行下一个example
            continue

        total += 1 # 满足条件的example + 1
        # 用0初始化所有的单词 --> 数字的矩阵
        context_idxs = np.zeros([para_limit], dtype=np.int32)
        context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
        y1 = np.zeros([para_limit], dtype=np.float32)
        y2 = np.zeros([para_limit], dtype=np.float32)

        def _get_word(word):
            # 对一个单词的lower(), capitalize(), upper()认为是不同的词，如果word本身找不到，就找word.lower()，以此类推
            # 都找不到，则返回1表示--OOV--
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1
        # 同理与_get_word()
        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        for i, token in enumerate(example["context_tokens"]):
            context_idxs[i] = _get_word(token)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idxs[i] = _get_word(token)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit: # 对于char长度大于char_limit的，后面的直接丢弃
                    break
                context_char_idxs[i, j] = _get_char(char)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[i, j] = _get_char(char)

        start, end = example["y1s"][-1], example["y2s"][-1]
        y1[start], y2[end] = 1.0, 1.0

        record = tf.train.Example(features=tf.train.Features(feature={
                                  "context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
                                  "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
                                  "context_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_char_idxs.tostring()])),
                                  "ques_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_idxs.tostring()])),
                                  "y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
                                  "y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
                                  "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))
                                  }))
        writer.write(record.SerializeToString())
    print("Built {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    # 将处理好的数据保存在TFRecords里，并返回共有多少条数据
    writer.close()
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def prepro(config):
    # 虽然precess_file()没有返回word_counter和char_counter，但两个字典都被更新了
    word_counter, char_counter = Counter(), Counter()
    train_examples, train_eval = process_file(
        config.train_file, "train", word_counter, char_counter)
    dev_examples, dev_eval = process_file(
        config.dev_file, "dev", word_counter, char_counter)
    test_examples, test_eval = process_file(
        config.test_file, "test", word_counter, char_counter)

    word_emb_file = config.fasttext_file if config.fasttext else config.glove_word_file
    char_emb_file = config.glove_char_file if config.pretrained_char else None
    char_emb_size = config.glove_char_size if config.pretrained_char else None
    char_emb_dim = config.glove_dim if config.pretrained_char else config.char_dim

    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, "word", emb_file=word_emb_file, size=config.glove_word_size, vec_size=config.glove_dim)
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, "char", emb_file=char_emb_file, size=char_emb_size, vec_size=char_emb_dim)

    build_features(config, train_examples, "train",
                   config.train_record_file, word2idx_dict, char2idx_dict)
    dev_meta = build_features(config, dev_examples, "dev",
                              config.dev_record_file, word2idx_dict, char2idx_dict)
    test_meta = build_features(config, test_examples, "test",
                               config.test_record_file, word2idx_dict, char2idx_dict, is_test=True)
    # 保存word/char emb矩阵 & word/char2idx字典；eval数据；meta数据
    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.char_emb_file, char_emb_mat, message="char embedding")
    save(config.train_eval_file, train_eval, message="train eval")
    save(config.dev_eval_file, dev_eval, message="dev eval")
    save(config.test_eval_file, test_eval, message="test eval")
    save(config.dev_meta, dev_meta, message="dev meta")
    save(config.test_meta, test_meta, message="test meta")
    save(config.word_dictionary, word2idx_dict, message="word dictionary")
    save(config.char_dictionary, char2idx_dict, message="char dictionary")
