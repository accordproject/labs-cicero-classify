import pandas as pd
import pymongo
import numpy as np

from core.config import MONGODB_URL,DATABASE_NAME, NER_LABEL_COLLECTION, LABEL_COLLECTION, LABEL_TRAIN_JOB_COLLECTION, CONFIG_COLLECTION, NER_TRAINER_DATA_CATCH_FILE, NER_ADAPTERS_TRAINER_NAME

from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import OneHotEncoder


import numpy as np

from tqdm import tqdm

import swifter
import os

client = pymongo.MongoClient(MONGODB_URL)
def get_training_dataframe(train_data_search_filter = {}, cache = True):
    try: #load from cache
        client = pymongo.MongoClient(MONGODB_URL)
        config_col = client[DATABASE_NAME][CONFIG_COLLECTION] 
        trainer = config_col.find_one({
            "name": NER_ADAPTERS_TRAINER_NAME
        })
        train_dataset = config_col.find_one({
            "collection_name": NER_LABEL_COLLECTION
        }) 
        if cache:
            if train_dataset["last_update_time"] == trainer["last_data_cache_timestamp"]:
                print("Reading Data from Cache...")
                df = pd.read_csv(trainer["data_cache_path"])
                df["labels"] = df["cache_labels"].swifter.progress_bar(False).apply(lambda x: x.split("|"))
                del df["cache_labels"]
                return df
            else: # new update, no cache
                os.remove(NER_TRAINER_DATA_CATCH_FILE)
    except Exception as e:
        print(e)
        pass

    col = client[DATABASE_NAME][NER_LABEL_COLLECTION]

    result = col.find(train_data_search_filter)
    print("Reading Data from MongoDB...")
    dfs = []
    df_columns = ["Sentence #", "token", "labels"]
    for i, sentence in enumerate(tqdm(result, total = col.count_documents(train_data_search_filter))):
        if i%50 == 0:
            if i != 0: dfs.append(df)
            df = pd.DataFrame()
        sentense_df = pd.DataFrame(columns = df_columns, data = sentence["token_and_labels"])
        sentense_df["Sentence #"] = str(sentence["_id"])
        df = df.append(sentense_df)
    dfs.append(df)
    print("Loading Data to DataFrame")
    while len(dfs) > 50:
        new_dfs = []
        tmp_df = pd.DataFrame(columns = df_columns)
        for i, df in enumerate((dfs)):
            if i%50 == 0:
                if i != 0: new_dfs.append(tmp_df)
                tmp_df = pd.DataFrame()
            tmp_df = tmp_df.append(df)
        new_dfs.append(tmp_df)
        dfs = new_dfs
    final_df = pd.DataFrame(columns = df_columns)
    for df in tqdm(dfs):
        final_df = final_df.append(df)
    final_df = final_df.reset_index(drop=True)
    # Cache
    final_df["cache_labels"] = final_df["labels"].swifter.progress_bar(False).apply(lambda x: "|".join(x))
    final_df.to_csv(NER_TRAINER_DATA_CATCH_FILE, index=False,
                    columns = df_columns - ["labels"] + ["cache_labels"])
    del final_df["cache_labels"]
    config_col.update_one({
        "name": NER_ADAPTERS_TRAINER_NAME
    }, {
        "$set": {
            "last_data_cache_timestamp": train_dataset["last_update_time"],
            "data_cache_path": NER_TRAINER_DATA_CATCH_FILE,
            }
    })
    return final_df

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

    sentences = df.groupby("Sentence #")["token"].apply(list).values
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

