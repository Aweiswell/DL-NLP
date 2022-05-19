import gensim.models as w2v
import jieba

jieba.setLogLevel(jieba.logging.INFO)

seg_novel = []
stop_words = []
dict_words = []

stop_words_file = open("stop_words.txt", 'r', encoding='utf-8')
stop_words = list()
for line in stop_words_file.readlines():
    line = line.strip()   # 去掉每行末尾的换行符
    stop_words.append(line)
stop_words_file.close()
print(f'{len(stop_words)} stop_words loaded')

dict_words_file = open("三体词典.txt", 'r', encoding='utf-8')
dict_words = list()
for line in dict_words_file.readlines():
    line = line.strip()   # 去掉每行末尾的换行符
    jieba.add_word(line)
    dict_words.append(line)
stop_words_file.close()
print(f'{len(dict_words)} dict_words loaded')


novel_name = '三体.txt'
novel = open(novel_name, 'r', encoding='utf-8')

print(f'Processing {novel_name}...')

line = novel.readline()
while line:
    line_1 = line.strip()
    outstr = ''
    line_seg = jieba.cut(line_1, cut_all=False)
    for word in line_seg:
        if word not in stop_words:
            if word != '\t':
                if word[:2] in dict_words:
                    word = word[:2]
                outstr += word
                outstr += " "
    if len(str(outstr.strip())) != 0:
        seg_novel.append(str(outstr.strip()).split())
    line = novel.readline()

print(f'{novel_name} finished，with {len(seg_novel)} row')
# print("-" * 40)

# one line = one sentence. Words must be already preprocessed and separated by whitespace.
# sentences:句子，min_count:词频小于设定值的词扔掉，window:一次取的词数，vector_size:词向量的维度.
# sg ({0, 1}, optional)
# 1 for skip-gram; 0 for CBOW.
# hs ({0, 1}, optional)
# If 1, hierarchical softmax will be used for model training.
# If 0, and negative is non-zero, negative sampling will be used.

# 保存（也可以保存为txt）、使用和输出
# model.wv.save_word2vec_format("WV.model_128.bin", binary=True)
# model = w2v.KeyedVectors.load_word2vec_format("WV.model.bin", binary=True)
# vector = model.word_vec['computer']

model = w2v.Word2Vec(sentences=seg_novel, min_count=30, window=5, vector_size=256, sg=0, hs=0, negative=5)
# model.save('CBOW.model')
model.wv.save_word2vec_format('CBOW.txt', binary=False)

model = w2v.Word2Vec(sentences=seg_novel, min_count=30, window=5, vector_size=256, sg=1, hs=0, negative=5)
# model.save('SkipGram.model')
model.wv.save_word2vec_format('SkipGram.txt', binary=False)
