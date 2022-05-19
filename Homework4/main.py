import gensim.models as w2v


def find_relation(a, b, c):
    d, _ = model.most_similar(positive=[c, b], negative=[a])[0]
    print(f' {d} 之于 {c} 相当于 {b} 之于 {a} ')
    print()


def cal_similarity(a, b):
    print(f' {a} 与 {b} 的相关程度为： {model.similarity(a, b)} ')
    print()


def find_top(a, x):
    # x = 10
    topx = model.most_similar(a, topn=x)
    print(f'与 {a} 最相关的前 {x} 个词为:')
    for i in range(x):
        print(topx[i])
    print()


def show_vec(a):
    print(f' {a} 的词向量为:')
    print(model.get_vector(a))


model = w2v.KeyedVectors.load_word2vec_format('SkipGram.txt', binary=False)
# model = w2v.KeyedVectors.load_word2vec_format('CBOW.txt', binary=False)

# show_vec('汪淼')
# show_vec('罗辑')

find_top('汪淼', 10)
cal_similarity('汪淼', '史强')
cal_similarity('汪淼', '大史')
cal_similarity('汪淼', '叶文洁')
cal_similarity('汪淼', '云天明')
cal_similarity('汪淼', '章北海')

find_top('罗辑', 10)
cal_similarity('罗辑', '史强')
cal_similarity('罗辑', '大史')
cal_similarity('罗辑', '庄颜')
cal_similarity('罗辑', '云天明')
cal_similarity('罗辑', '章北海')

find_top('叶文洁', 10)
find_top('程心', 10)
find_top('章北海', 10)

cal_similarity('大史', '史强')
find_top('大史', 10)
find_top('史强', 10)

find_relation('罗辑', '大史', '罗辑')
