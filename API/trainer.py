try:
    import pandas as pd
    import pymongo
    import numpy as np
    import torch

    from core.config import (
        MONGODB_URL,
        DATABASE_NAME,
        Feedback_Label_Collection,
        LABEL_COLLECTION,
        LABEL_RETRAIN_QUEUE_COLLECTION,
        NER_TRAIN_BATCH_SIZE,
        NER_TRAIN_DEFAULT_FILTER,
        NER_TRAIN_DEVIDE_ID,
        NER_ADAPTERS_PATH
    )
    from torch.utils.data import Dataset
    from sklearn.preprocessing import OneHotEncoder

    from utils.trainer.NER import get_training_dataframe, NER_Dataset_for_Adapter
    import re
    import sys

    import os
    import datetime
    # When Each Train
    # Run When Set Up
    if os.path.isdir(f"{NER_ADAPTERS_PATH}/save_adapters") == False:
        os.mkdir(f"{NER_ADAPTERS_PATH}/save_adapters")
    if os.path.isdir(f"{NER_ADAPTERS_PATH}/save_heads") == False:
        os.mkdir(f"{NER_ADAPTERS_PATH}/save_heads")

    dateStamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")

    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    device = torch.device(f"cuda:{NER_TRAIN_DEVIDE_ID}" if torch.cuda.is_available() else "cpu")


    #df = get_training_dataframe(NER_TRAIN_DEFAULT_FILTER)
    df = pd.read_csv("test_csv_df.csv")

    def get_target_df_by_filter(df, train_data_search_filter):
        client = pymongo.MongoClient(MONGODB_URL)
        col = client[DATABASE_NAME][Feedback_Label_Collection]

        wanted_id = list(map(lambda x: str(x["_id"]),
                             list(col.find(train_data_search_filter, {"_id": True}))))

        target_df = df[df["Sentence #"].isin(wanted_id)]
        return target_df

    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_sequence

    def create_mini_batch(samples):
        tokens_tensors = [s[0] for s in samples]
        segments_tensors = [s[1] for s in samples]

        # 
        if samples[0][2] is not None:
            label_ids = [s[2] for s in samples]
            label_ids = pad_sequence(label_ids, 
                                      batch_first=True)
        else:
            label_ids = None

        # zero pad to same length
        tokens_tensors = pad_sequence(tokens_tensors, 
                                      batch_first=True)
        segments_tensors = pad_sequence(segments_tensors, 
                                        batch_first=True)

        # attention masks, set zero padding in tokens_tensors
        # to 1 so BERT only attention on those tokens
        masks_tensors = torch.zeros(tokens_tensors.shape, 
                                    dtype=torch.long)
        masks_tensors = masks_tensors.masked_fill(
            tokens_tensors != 0, 1)

        return tokens_tensors, segments_tensors, masks_tensors, label_ids


    from utils.logs import trainer_log, queue_task_log

    client = pymongo.MongoClient(MONGODB_URL)
    training_queue_col = client[DATABASE_NAME][LABEL_RETRAIN_QUEUE_COLLECTION]

    while True:
        training_queue = training_queue_col.find({
            "status":  re.compile("(training|waiting)")
        })
        training_queue = list(training_queue)

        if len(training_queue) == 0:
            break

        # Old Train First
        training_queue.sort(key = lambda x: x["add_time"], reverse= False)

        now_is_training = training_queue[0]

        # Update parameter On Each Iter
        train_data_search_filter = now_is_training["train_data_filter"]
        label_name = now_is_training["label_name"]
        Epoch_Times = now_is_training["epochs"]

        target_df = get_target_df_by_filter(df, train_data_search_filter)
        trainset = NER_Dataset_for_Adapter(tokenizer, target_df, label_name)

        log_msg = f"Start training {label_name}, have {len(training_queue) -1} in the waiting line..."
        trainer_log(log_msg)
        queue_task_log(now_is_training["_id"], log_msg)

        training_queue_col.update_one({
                "_id": now_is_training["_id"],
            },{
                "$set": {
                    "status": "training",
                    "train_data_count": len(trainset)
                }
            })

        label_define_col = client[DATABASE_NAME][LABEL_COLLECTION]
        label_define_col.update_one(
            {"label_name": label_name},
            {"$set": {"adapter.current_adapter_filename": f"{label_name}_epoch_{Epoch_Times}_{dateStamp}",
                    "adapter.training_status": "training a new one",
            }})

        trainloader = DataLoader(trainset, batch_size=NER_TRAIN_BATCH_SIZE, 
                                 collate_fn=create_mini_batch)

        from transformers import RobertaConfig, RobertaModelWithHeads
        config = RobertaConfig.from_pretrained(
            "roberta-base"
        )

        try:
            model = RobertaModelWithHeads.from_pretrained(
                "roberta-base",
                config=config,
                )

            try:
                model.add_adapter(label_name)
                model.add_tagging_head(
                    label_name,
                    num_labels=1
                  )
            except: pass
            model.train_adapter(label_name)
            model = model.to(device)

            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                            {
                                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                                "weight_decay": 1e-5,
                            },
                            {
                                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                                "weight_decay": 0.0,
                            },
                        ]
            optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=1e-4)

            for epoch in range(Epoch_Times):
                epoch += 1 #start from 1
                print(f"\n{label_name}: epoch {epoch}")
                for i, data in enumerate(trainloader):

                    tokens_tensors, segments_tensors, \
                    masks_tensors, labels = [t.to(device) for t in data]

                    outputs = model(input_ids = tokens_tensors,
                        attention_mask=masks_tensors,
                        token_type_ids=segments_tensors)


                    logits = outputs[0]

                    current_label = labels.view(-1, labels.shape[-1])[:, trainset.label_map[label_name]]
                    current_label = current_label.view(-1)

                    active_logits = logits.view(-1, logits.shape[-1])[masks_tensors.view(-1) == 1]
                    active_labels = current_label[masks_tensors.view(-1)== 1]

                    actual = current_label[masks_tensors.view(-1)== 1].float().view(-1,1)

                    loss_fct = torch.nn.BCEWithLogitsLoss()

                    loss = loss_fct(active_logits, actual)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    if i % 100 == 0:
                        print(f"\tLoss: {loss}")
                        queue_task_log(now_is_training["_id"], f"[At Epoch {epoch} Round {i}] Loss: {loss}")
                """filename = f"{label_name}_epoch_{epoch}_{dateStamp}"
                model.save_adapter(f"{NER_ADAPTERS_PATH}/save_adapters/{filename}", model.active_adapters[0])
                model.save_head(f"{NER_ADAPTERS_PATH}/save_heads/{filename}", model.active_head)"""
            filename = f"{label_name}_epoch_{Epoch_Times}_{dateStamp}"
            model.save_adapter(f"{NER_ADAPTERS_PATH}/save_adapters/{filename}", model.active_adapters[0])
            model.save_head(f"{NER_ADAPTERS_PATH}/save_heads/{filename}", model.active_head)
        except Exception as error:
            trainer_log(error.args[0])
            queue_task_log(now_is_training["_id"], log_msg)
            if "CUDA out of memory" in error.args[0]:
                sys.exit(4)
            else:
                raise error

        training_queue_col.update_one({
                "_id": now_is_training["_id"],
            },{
                "$set": {
                    "store_filename": f"{label_name}_epoch_{Epoch_Times}_{dateStamp}",
                    "status": "done",
                }
            })

        label_define_col = client[DATABASE_NAME][LABEL_COLLECTION]
        now_time = datetime.datetime.now()
        label_define_col.update_one(
            {"label_name": label_name},
            {"$set": {"adapter.current_adapter_filename": f"{label_name}_epoch_{Epoch_Times}_{dateStamp}",
                      "adapter.training_status": "done",
                      "adapter.update_time": now_time,
                },
             "$push": {"adapter.history": {
                    "adapter_filename": f"{label_name}_epoch_{Epoch_Times}_{dateStamp}",
                    "time": now_time,
                }}})
except KeyboardInterrupt:
    pass