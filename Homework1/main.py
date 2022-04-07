# coding=gbk
import os
import re
import math
import jieba


def text_preprocessing(tmp_text_path):
    file_dir = os.listdir(tmp_text_path)
    char_to_be_replaced = '[a-zA-Z0-9’!"#$%&\'()（）*+,-./:：;；<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]'
    tmp_corpus = []
    tmp_char_num = 0
    for tmp_file_name in file_dir:
        path = os.path.join(tmp_text_path, tmp_file_name)
        if os.path.isfile(path):
            with open(path, "r", encoding="gbk", errors="ignore") as tmp_file:
                tmp_file_context = tmp_file.read()
                tmp_file_context = tmp_file_context.replace("本书来自www.cr173.com免费txt小说下载站", '')
                tmp_file_context = tmp_file_context.replace("更多更新免费电子书请关注www.cr173.com", '')
                tmp_file_context = re.sub(char_to_be_replaced, '', tmp_file_context)
                tmp_file_context = tmp_file_context.replace("\n", '')
                tmp_file_context = tmp_file_context.replace(" ", '')
                tmp_file_context = tmp_file_context.replace("\u3000", '')
                tmp_char_num += len(tmp_file_context)
                tmp_corpus.append(tmp_file_context)
    return tmp_corpus, tmp_char_num


# 统计词频
# 根据词库words创建一元词频词典tf_dic1
def get_tf(tf_dic1, words):
    for i in range(len(words) - 1):
        tf_dic1[words[i]] = tf_dic1.get(words[i], 0) + 1


# 根据词库words创建二元词频词典tf_dic2
def get_bigram_tf(tf_dic2, words):
    for i in range(len(words) - 1):
        tf_dic2[(words[i], words[i + 1])] = tf_dic2.get((words[i], words[i + 1]), 0) + 1


# 根据词库words创建三元词频词典tf_dic3
def get_trigram_tf(tf_dic3, words):
    for i in range(len(words) - 2):
        tf_dic3[((words[i], words[i + 1]), words[i + 2])] = tf_dic3.get(((words[i], words[i + 1]), words[i + 2]), 0) + 1


def cal_char_1gram(tmp_corpus):
    split_words = []
    words_len = 0
    words_tf = {}
    for tmp_text in tmp_corpus:
        for char in tmp_text:
            split_words.append(char)
            words_len += 1
        get_tf(words_tf, split_words)
        split_words = []
    entropy = []
    for uni_word in words_tf.items():
        entropy.append(-(uni_word[1] / words_len) * math.log(uni_word[1] / words_len, 2))
    print("一元模型信息熵:", round(sum(entropy), 3), "比特/字")


def cal_char_2gram(tmp_corpus):
    split_words = []
    words_tf = {}
    bigram_tf = {}
    for tmp_text in tmp_corpus:
        for char in tmp_text:
            split_words.append(char)
        get_tf(words_tf, split_words)
        get_bigram_tf(bigram_tf, split_words)
        split_words = []
    bigram_len = sum([dic[1] for dic in bigram_tf.items()])
    entropy = []
    for bi_word in bigram_tf.items():
        jp_xy = bi_word[1] / bigram_len  # p(x,y)
        cp_xy = bi_word[1] / words_tf[bi_word[0][0]]  # p(x|y)
        entropy.append(-jp_xy * math.log(cp_xy, 2))  # 计算信息熵
    print("二元模型信息熵:", round(sum(entropy), 3), "比特/字")


def cal_char_3gram(tmp_corpus):
    split_words = []
    words_tf = {}
    trigram_tf = {}
    for tmp_text in tmp_corpus:
        for char in tmp_text:
            split_words.append(char)
        get_bigram_tf(words_tf, split_words)
        get_trigram_tf(trigram_tf, split_words)
        split_words = []
    trigram_len = sum([dic[1] for dic in trigram_tf.items()])
    entropy = []
    for tri_word in trigram_tf.items():
        jp_xy = tri_word[1] / trigram_len  # p(x,y)
        cp_xy = tri_word[1] / words_tf[tri_word[0][0]]  # p(x|y)
        entropy.append(-jp_xy * math.log(cp_xy, 2))  # 计算信息熵
    print("三元模型信息熵:", round(sum(entropy), 3), "比特/字")


def cal_word_1gram(tmp_corpus):
    split_words = []
    words_len = 0
    words_tf = {}
    for tmp_text in tmp_corpus:
        for word in jieba.cut(tmp_text):
            split_words.append(word)  # 分词
            words_len += 1
        get_tf(words_tf, split_words)
        split_words = []
    print("分词个数:", words_len)
    print("平均词长:", round(char_num / words_len, 3))
    entropy = []
    for uni_word in words_tf.items():
        entropy.append(-(uni_word[1] / words_len) * math.log(uni_word[1] / words_len, 2))
    print("一元模型信息熵:", round(sum(entropy), 3), "比特/词")


def cal_word_2gram(tmp_corpus):
    split_words = []
    words_tf = {}
    bigram_tf = {}
    for tmp_text in tmp_corpus:
        for word in jieba.cut(tmp_text):
            split_words.append(word)  # 分词
        get_tf(words_tf, split_words)
        get_bigram_tf(bigram_tf, split_words)
        split_words = []
    bigram_len = sum([dic[1] for dic in bigram_tf.items()])
    entropy = []
    for bi_word in bigram_tf.items():
        jp_xy = bi_word[1] / bigram_len  # p(x,y)
        cp_xy = bi_word[1] / words_tf[bi_word[0][0]]  # p(x|y)
        entropy.append(-jp_xy * math.log(cp_xy, 2))  # 计算信息熵
    print("二元模型信息熵:", round(sum(entropy), 3), "比特/词")


def cal_word_3gram(tmp_corpus):
    split_words = []
    words_tf = {}
    trigram_tf = {}
    for tmp_text in tmp_corpus:
        for word in jieba.cut(tmp_text):
            split_words.append(word)  # 分词
        get_bigram_tf(words_tf, split_words)
        get_trigram_tf(trigram_tf, split_words)
        split_words = []
    trigram_len = sum([dic[1] for dic in trigram_tf.items()])
    entropy = []
    for tri_word in trigram_tf.items():
        jp_xy = tri_word[1] / trigram_len  # p(x,y)
        cp_xy = tri_word[1] / words_tf[tri_word[0][0]]  # p(x|y)
        entropy.append(-jp_xy * math.log(cp_xy, 2))  # 计算信息熵
    print("三元模型信息熵:", round(sum(entropy), 3), "比特/词")


if __name__ == '__main__':
    text_path = './jyxstxtqj_downcc.com/'
    corpus, char_num = text_preprocessing(text_path)
    print("— — — — — — 以字为单位 — — — — — —")
    print("汉字个数:", char_num)
    cal_char_1gram(corpus)
    cal_char_2gram(corpus)
    cal_char_3gram(corpus)
    jieba.setLogLevel(jieba.logging.INFO)  # 隐藏jieba运行结果
    print("— — — — — — 以词为单位 — — — — — —")
    cal_word_1gram(corpus)
    cal_word_2gram(corpus)
    cal_word_3gram(corpus)
