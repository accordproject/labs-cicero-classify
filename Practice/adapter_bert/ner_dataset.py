import pandas as pd
import numpy as np
import torch

data_path = "./NER_multilabel_data_v2.csv"
BATCH_SIZE = 16
def get_trainset_data_loader(tokenizer, data_path = data_path,
                             BATCH_SIZE = BATCH_SIZE):
    df = pd.read_csv(data_path)

    all_tags = df.newTag

    all_tags = set(all_tags)

    all_tags = "|".join(all_tags)
    all_tags = all_tags.split("|")
    all_tags = set(all_tags)
    all_tags = list(all_tags)


    def process_csv(data_path):
        df = pd.read_csv(data_path, encoding="latin-1")
        df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")
        sentences = df.groupby("Sentence #")["Word"].apply(list).values
        tags = df.groupby("Sentence #")["newTag"].apply(list).values
        return sentences, tags

    sentences, tags = process_csv(data_path)

    from torch.utils.data import Dataset
    from sklearn.preprocessing import OneHotEncoder




    class NER_Dataset(Dataset):
        # 讀取前處理後的 tsv 檔並初始化一些參數
        def __init__(self, mode, tokenizer, data_path, labels):
            assert mode in ["train", "test"]  # 一般訓練你會需要 dev set
            self.mode = mode
            # 大數據你會需要用 iterator=True
            self.sentences, self.tags = process_csv(data_path)
            self.len = len(self.sentences)


            if mode != "test":
                self.label_map = {}
                for i in range(len(labels)):
                    self.label_map[labels[i]] = i

                possible_labels = np.array(range(len(labels))).reshape(-1, 1)
                self.oneHotEncoder = OneHotEncoder()
                self.oneHotEncoder.fit(possible_labels)
            else:
                self.label_map = None

            self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer
            self.O_label = self.label_map["O"]


        # 定義回傳一筆訓練 / 測試數據的函式
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
            word_pieces = ['[CLS]']
            word_pieces += self.sentences[idx]
            word_pieces += ['[SEP]']

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


    # 初始化一個專門讀取訓練樣本的 Dataset，使用中文 BERT 斷詞

    df = pd.read_csv(data_path, encoding="latin-1")

    labels = np.unique("|".join(list(df.newTag)).split("|"))
    print(f"labels: {labels}")

    trainset = NER_Dataset("train", tokenizer=tokenizer, data_path=data_path, labels= labels)

    from torch.utils.data import DataLoader, IterableDataset
    from torch.nn.utils.rnn import pad_sequence
    def create_mini_batch(samples):
        tokens_tensors = [s[0] for s in samples]
        segments_tensors = [s[1] for s in samples]

        # 測試集有 labels
        if samples[0][2] is not None:
            label_ids = [s[2] for s in samples]
            label_ids = pad_sequence(label_ids, 
                                      batch_first=True)
        else:
            label_ids = None

        # zero pad 到同一序列長度
        tokens_tensors = pad_sequence(tokens_tensors, 
                                      batch_first=True)
        segments_tensors = pad_sequence(segments_tensors, 
                                        batch_first=True)

        # attention masks，將 tokens_tensors 裡頭不為 zero padding
        # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
        masks_tensors = torch.zeros(tokens_tensors.shape, 
                                    dtype=torch.long)
        masks_tensors = masks_tensors.masked_fill(
            tokens_tensors != 0, 1)

        return tokens_tensors, segments_tensors, masks_tensors, label_ids





    trainset.id2label = {}
    for key in trainset.label_map.keys():
        trainset.id2label[trainset.label_map[key]] = key


    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, 
                             collate_fn=create_mini_batch)

    data = next(iter(trainloader))

    tokens_tensors, segments_tensors, \
        masks_tensors, label_ids = data

    '''print(f"""
    tokens_tensors.shape   = {tokens_tensors.shape} 
    {tokens_tensors}
    ------------------------
    segments_tensors.shape = {segments_tensors.shape}
    {segments_tensors}
    ------------------------
    masks_tensors.shape    = {masks_tensors.shape}
    {masks_tensors}
    ------------------------
    label_ids.shape        = {label_ids.shape}
    {label_ids}
    """)'''
    
    trainset.id2label = {}
    for key in trainset.label_map.keys():
        trainset.id2label[trainset.label_map[key]] = key

    
    return all_tags, trainset, trainloader