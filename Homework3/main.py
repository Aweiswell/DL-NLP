# coding=gbk
# https://blog.csdn.net/weixin_42663984/article/details/116264233

import jieba
import numpy as np
import os
import re
from gensim import corpora, models


def get_data():
    outfilename_1 = "./train.txt"
    outfilename_2 = "./test.txt"

    if not os.path.exists('./train.txt'):
        outputs = open(outfilename_1, 'w', encoding='gbk')
        outputs_test = open(outfilename_2, 'w', encoding='gbk')
        datasets_root = "./jyxstxtqj_downcc.com"
        # catalog = "in.txt"
        catalog = "inf.txt"

        test_num = 50 * 13  # ��������
        # test_length = 20  # ���䳤��

        with open(os.path.join(datasets_root, catalog), "r", encoding='gbk') as f:
            all_files = f.readline().split(",")
            print(all_files)

        for name in all_files:
            with open(os.path.join(datasets_root, name + ".txt"), "r", encoding='gbk',
                      errors="ignore") as f:
                file_read = f.readlines()
                train_num = len(file_read) - test_num
                choice_index = np.random.choice(len(file_read), test_num + train_num, replace=False)
                char_to_be_replaced = '[a-zA-Z0-9��!"#$%&\'()����*+,-./:��;��<=>?@����?�����������������������[\\]^_`{|}~]'
                word_to_be_replaced = ['��', '��', '��', '��', '��', '��', '��', '��', '��', 'Ҳ', '�', '��', '��', 'ȴ',
                                       '��', '��', '��', '˵', 'Ի', '��', '��', '��', 'Ϊ', '֮', '��', '��', '��', 'ͬ',
                                       '��', 'ȥ', '��', '��', '��', 'Ҫ', '��', 'ô', '��', '��', '��', '��', 'ֻ', '��',
                                       'ȥ', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��',
                                       '��', '��', '��', '��', '��', '��', '��', 'һ', '��', 'û', '˭', '��', '��', '��',
                                       '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', 'ʱ', '��',
                                       '��', '��', 'ʲ', '��', '��', '��', '��', 'Ц', '��', '��',
                                       '�Լ�']
                for train in choice_index[0:train_num]:
                    line = file_read[train]
                    line = re.sub(char_to_be_replaced, '', line)
                    for word in word_to_be_replaced:
                        line = re.sub(word, '', line)
                    line = re.sub('[\u0000-\u4DFF]', '', line)
                    line = re.sub('[\u9FA6-\uFFFF]', '', line)
                    if len(line) == 0:
                        continue
                    seg_list = list(jieba.cut(line, cut_all=False))
                    line_seg = ""
                    for term in seg_list:
                        line_seg += term + " "
                        # for index in range len(line_seg):
                    outputs.write(line_seg.strip() + '\n')

                count = 0
                for test in choice_index[train_num:test_num + train_num]:
                    # if test + test_length >= len(file_read):
                    #     continue
                    test_line = ""
                    # for i in range(test, test + test_length):
                    line = file_read[test]
                    line = re.sub(char_to_be_replaced, '', line)
                    for word in word_to_be_replaced:
                        line = re.sub(word, '', line)
                    line = re.sub('[\u0000-\u4DFF]', '', line)
                    line = re.sub('[\u9FA6-\uFFFF]', '', line)
                    line = re.sub('\n', '', line)
                    # if len(line) == 0:
                    #     continue
                    seg_list = list(jieba.cut(line, cut_all=False))
                    # line_seg = ""
                    for term in seg_list:
                        test_line += term + " "
                    outputs_test.write(test_line.strip())
                    count += 1
                    if count == 50:
                        outputs_test.write('\n')
                        count = 0
                outputs_test.write('\n')

        outputs.close()
        outputs_test.close()
        print("�õ�ѵ���Ͳ����ı�")


if __name__ == "__main__":
    get_data()

    fr = open('./train.txt', 'r', encoding='gbk')
    train = []

    for line in fr.readlines():
        line = [word.strip() for word in line.split(' ')]
        train.append(line)

    """ѵ��LDAģ��"""
    dictionary = corpora.Dictionary(train)

    corpus = [dictionary.doc2bow(text) for text in train]

    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=16)

    topic_list_lda = lda.print_topics(16)

    print("��LDAΪ��������16������ĵ��ʷֲ�Ϊ��\n")
    for topic in topic_list_lda:
        print(topic)

    file_test = "./test.txt"
    news_test = open(file_test, 'r', encoding='gbk')
    test = []

    for line in news_test:
        line = [word.strip() for word in line.split(' ')]
        test.append(line)

    for text in test:
        corpus_test = dictionary.doc2bow(text)

    corpus_test = [dictionary.doc2bow(text) for text in test]

    topics_test = lda.get_document_topics(corpus_test)

    for i in range(200):
        print(i)
        print('������ֲ�Ϊ��\n')
        print(topics_test[i], '\n')

    fr.close()
    news_test.close()
