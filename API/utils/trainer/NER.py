import pandas as pd
import pymongo
import numpy as np

from core.config import MONGODB_URL,DATABASE_NAME, Feedback_Label_Collection, LABEL_COLLECTION, LABEL_RETRAIN_QUEUE_COLLECTION

from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import OneHotEncoder


import numpy as np

from tqdm import tqdm

client = pymongo.MongoClient(MONGODB_URL)
def get_training_dataframe(train_data_search_filter = {}):
    col = client[DATABASE_NAME][Feedback_Label_Collection]

    result = col.find(train_data_search_filter)
    print("Reading Data from MongoDB...")
    df = pd.DataFrame()
    for i, sentence in enumerate(tqdm(result, total = col.count_documents(train_data_search_filter))):
        sentense_df = pd.DataFrame(columns=["Sentence #", "text", "labels"], data = sentence["text_and_labels"])
        sentense_df["Sentence #"] = str(sentence["_id"])
        df = df.append(sentense_df)

def get_training_data_by_df_according_to_label_name(df, label_name):
    label_col = client[DATABASE_NAME][LABEL_COLLECTION]

    label_info = label_col.find_one({"label_name": label_name})
    alias_labels = label_col.find({"alias_as": {"$in": [label_name]}})

    alias = []
    for alias_label in alias_labels:
        alias.append(alias_label["label_name"])
    alias = alias + label_info["inherit"]

    wanted_label = [label_name] + alias

    def label_data(label):
        if set(label).intersection(set(wanted_label)):
            return label_name
        else:
            return "O"

    df[label_name] = list(df["labels"].apply(label_data))

    sentences = df.groupby("Sentence #")["text"].apply(list).values
    tags = df.groupby("Sentence #")[label_name].apply(list).values
    return sentences, tags

class NER_Dataset_for_Adapter(Dataset):
    def __init__(self, tokenizer, df, label_name):
        self.label_name = label_name
        self.mode = "train"
        # 大數據你會需要用 iterator=True
        self.sentences, self.tags = get_training_data_by_df_according_to_label_name(df, label_name)
        self.len = len(self.sentences)


        labels = ["O", label_name]

        if self.mode != "test":
            labels = ["O", label_name]
            self.label_map = {}
            for i, label in enumerate(labels):
                self.label_map[label] = i

            possible_labels = np.array(range(len(labels))).reshape(-1, 1)
            self.oneHotEncoder = OneHotEncoder()
            self.oneHotEncoder.fit(possible_labels)
        else:
            self.label_map = None

        self.tokenizer = tokenizer  # RoBERTa tokenizer
        self.O_label = self.label_map["O"]

    def __getitem__(self, idx):
        if self.mode == "test":
            label_tensor = None
        else:
            label = ["O"] + self.tags[idx] + ["O"]

            label = np.array(label)
            label = label.reshape(-1,1)

            label = np.apply_along_axis(self.split_one_hot_multiTags, 1, label)
            label_tensor = torch.tensor(label, dtype = torch.float32)

        # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
        word_pieces = [self.tokenizer.cls_token]
        word_pieces += self.sentences[idx]
        word_pieces += [self.tokenizer.sep_token]

        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)

        # 將第一句包含 [SEP] 的 token 位置設為 0
        segments_tensor = torch.zeros_like(tokens_tensor)

        return (tokens_tensor, segments_tensor, label_tensor)

    def __len__(self):
        return self.len

    def split_one_hot_multiTags(self, tags):
        # tags = ['B-org|Party|String']
        tags = tags[0]
        tags = tags.split("|")


        tags_num = list(map(lambda x: self.label_map[x], tags))
        #[5, 20, 23]

        tags_num = np.array(tags_num).reshape(-1,1)

        tags_one_hot = self.oneHotEncoder.transform(tags_num).toarray()

        tags_one_hot = tags_one_hot.sum(axis = 0)

        #return torch.tensor(tags_one_hot, dtype = torch.float32)

        return tags_one_hot

