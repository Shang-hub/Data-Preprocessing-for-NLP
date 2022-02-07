import json
import os
from random import sample
import re, string
import pandas as pd



from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from torchkeras import Model


basic = os.getcwd()
dataset = basic + '/data/imdb/'

train_data_path = dataset + 'train.tsv'
test_data_path = dataset + 'test.tsv'

train_token_path = dataset + 'train_token.tsv'
test_token_path = dataset + 'test_token.tsv'

#Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MAX_WORDS = 10000 #考虑最高词频的MAX_WORDS个词
MAX_LEN = 200   #每个样本保留200个词的长度
BATCH_SIZE = 20

dftrain = pd.read_csv(train_data_path, encoding="utf-8", header=None, sep='\t') #对于没有表头的数据需要设置header=None
dftest = pd.read_csv(test_data_path, encoding="utf-8", header=None, sep="\t")

# print(dftrain.head(10)) #表示回复前10行的dataframe
# print(len(dftrain))
# print(len(dftest))
# print(dftrain[0].unique()) #dftrain[0]中不同的值

#demo
# s = "i love china i love nlp"
# d = {}
# #get的语法 dict.get(key, default=None)
# for word in s.split(' '):
#     d[word] = d.get(word,0) + 1
# print(d)


word_count_dict = {}

#清洗数据，将所有大写转换为小写，'\n' 替换为 ' '，'<br />' 替换为 ' ',去掉所有标点
def clean_text(text):
    lowercase = text.lower().replace("\n", ' ')
    cleaning_text = re.sub('<br />', ' ', lowercase)
    cleaned_text = re.sub("[%s]"%re.escape(string.punctuation), '', cleaning_text)
    return cleaned_text

#读入数据 清洗数据 构建词典  根据'\t'将文本分成列表，清洗数据，构建词典

with open(train_data_path, 'r', encoding='UTF-8') as f:
    for line in f:
        label, text = line.split('\t')
        cleaned_text = clean_text(text)
        for word in cleaned_text.split(' '):
            word_count_dict[word] = word_count_dict.get(word,0) + 1

# 字典转为dataframe, 只有一列, 列名count, 排序
df_word_dict = pd.DataFrame(pd.Series(word_count_dict, name = 'Count'))
df_word_dict = df_word_dict.sort_values(by='Count', ascending=False)
df_word_dict = df_word_dict[0:MAX_WORDS-2]
#print(df_word_dict)
df_word_dict['word_id'] = range(2,MAX_WORDS)
word_id_dict = df_word_dict['word_id'].to_dict()
id_word_dict = {v:k for k,v in word_id_dict.items()}

json.dump([word_id_dict, id_word_dict], open(dataset + 'word.json', 'w'))
word_id_dict, id_word_dict = json.load(open(dataset + 'word.json', 'r') )

#填充文本 如果超长，截取后pad_length长的文本 如果长度不够，从开头开始补1
def pad(data_list, pad_length):
    padded_list = data_list.copy()

    if len(data_list) > pad_length:
        padded_list = data_list[-pad_length:]
    
    if len(data_list) < pad_length:
        padded_list = [1] * (pad_length-len(data_list)) + data_list
    return padded_list

#打开即将写入的文件，分出来label 和 text，清洗文本，找到每个词的id，进行pad，label和id合成一起
def text_to_token(dfdata, token_file):
    with open(token_file, 'w', encoding='utf-8') as fout:
        for i in range(len(dfdata)):
            label = dfdata.loc[i, 0]
            text = dfdata.loc[i, 1]
            cleaned_text = clean_text(text)
            word_id_list = [word_id_dict.get(word, 0 ) for word in cleaned_text.split(' ')]
            #print("word_id_list")
            #print(word_id_list)
            padded_list = pad(word_id_list, MAX_LEN)
            out_line = str(label) + "\t" + " ".join([str(x) for x in padded_list])
            #print(out_line)
            fout.write(out_line + "\n")

text_to_token(dftrain, train_token_path)
text_to_token(dftest, test_token_path)

# Load text token file
class imdbDataset(Dataset):
    # 如果需要自定义Dataset，就需要实现__getitem__（）和 __len__（）方法。
    def __init__(self, tokenfile_path):
        self.tokenfile_path = tokenfile_path
        with open(self.tokenfile_path, 'r', encoding='UTF-8') as f:
            self.samples_list = []
            for count, line in enumerate(f, 1):
                self.samples_list.append(line)
            self.length = count

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        sample = self.samples_list[index]
        label, tokens = sample.split("\t")
        label = torch.tensor([float(label)], dtype=torch.float).to(device)
        feature = torch.tensor([int(x) for x in tokens.split(" ")], dtype=torch.long).to(device)
        return (feature, label)

ds_train = imdbDataset(train_token_path)
ds_test = imdbDataset(test_token_path)
dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True) #生成迭代器
dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)
# 输出shape大小
# for features, labels in ds_train:
#     print(features.shape)
#     print(labels)
#     break
# for features, labels in dl_train:
#     print(features.shape)   # (batchsize, 200)
#     print(labels.shape)
#     break

# Hyper params
H_SIZE = 64
N_LAYERS = 1
EMBEDDING_DIM = 3

class BiLSTM(nn.Module):
    def __init__(self, hidden_size=H_SIZE, num_layers=N_LAYERS):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_embeddings=MAX_WORDS, embedding_dim=EMBEDDING_DIM, padding_idx=1)
        self.lstm = nn.LSTM(EMBEDDING_DIM, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        x = self.embedding(x)  # (batch_size=20, MAX_LEN=200, embedding_dim=3)
        lstm_out, _ = self.lstm(x, (h0, c0))  # output shape: (batch_size=20, MAX_LEN=200, hidden_size*2=128)
        x_in = lstm_out[:, -1, :]  # shape: (20, 128)
        y = nn.Sigmoid()(self.fc(x_in))
        return y

def accuracy(y_pred, y_true):
    y_pred = torch.where(y_pred > 0.5,
                         torch.ones_like(y_pred, dtype=torch.float32),
                         torch.zeros_like(y_pred, dtype=torch.float32))
    acc = torch.mean(1-torch.abs(y_true-y_pred))
    return acc



model = BiLSTM()
model = Model(model)

model.compile(loss_func=nn.BCELoss(),
              optimizer=torch.optim.Adagrad(model.parameters(), lr = 0.02),
              metrics_dict={"accuracy":accuracy})

epochs = 20
dfhistory = model.fit(epochs, dl_train, dl_val=dl_test, log_step_freq=200)
print(dfhistory)

# 保存模型参数
torch.save(model.state_dict(),'./data/save_weight.pkl')

model.load_state_dict(torch.load('./data/save_weight.pkl'))
model.eval()
with torch.no_grad():
    file = model.predict(dl_test)
print(file)
