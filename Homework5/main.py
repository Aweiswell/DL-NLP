from textgenrnn import textgenrnn

def train():
    textgen = textgenrnn(name="novel") # 给模型起个名字

    textgen.reset() # 重置模型
    # 从数据文件训练模型
    textgen.train_from_file(file_path='./三体0.txt', # 文件路径
                            new_model=True, # 训练新模型
                            batch_size=4,
                            rnn_bidirectional=True, # 是否使用Bi-LSTM
                            rnn_size=64,
                            word_level=False, # True:词级别，False:字级别
                            dim_embeddings=300,
                            num_epochs=50, # 训练轮数
                            max_length=25, # 一条数据的最大长度
                            verbose=1)

    print(textgen.model.summary())

def textgen():
    textgen_2 = textgenrnn(weights_path='novel_weights.hdf5',
                           vocab_path='novel_vocab.json',
                           config_path='novel_config.json')

    textgen_2.generate_samples()


if __name__=="__main__":
    train()
    textgen()
