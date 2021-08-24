import pandas as pd
import numpy as np
import torch
print(f"Torch Version: {torch.__version__}")

import transformers
print(f"Transformers (Adapter) Version: {transformers.__version__}")

print(f"Loading adapter model...")

import numpy as np



from transformers import RobertaTokenizer
from utils.tokenizer import tokenizer

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")

from utils.ner_dataset import get_trainset_data_loader

from transformers import RobertaConfig, RobertaModelWithHeads

from utils.logger.utils import mute_logging
with mute_logging():
    config = RobertaConfig.from_pretrained("roberta-base")
    model = RobertaModelWithHeads.from_pretrained(
        "roberta-base",
        config=config,
    )

import os

# This is a asyncio status, but just use mongo client directly to save develop time
from pymongo import MongoClient
from core.config import MONGODB_URL, DATABASE_NAME, LABEL_COLLECTION, NER_ADAPTERS_PATH
mongo_client = MongoClient(MONGODB_URL)
labels_col = mongo_client[DATABASE_NAME][LABEL_COLLECTION]

labels = labels_col.find()

labels = list(labels)

all_adapters = []

def check_adapter_filename_valid(filename):
    if (os.path.isdir(f"{NER_ADAPTERS_PATH}/save_adapters/{filename}") and
        os.path.isdir(f"{NER_ADAPTERS_PATH}/save_heads/{filename}")):
        return True
    else:
        return False

for label in labels:
    if label["adapter"]["lastest_filename"]:
        filename = label["adapter"]["lastest_filename"]
        if check_adapter_filename_valid(filename) == False:
            while len(label["adapter"]["history"]) > 0:
                hisotry_adapter = label["adapter"]["history"].pop(0)
                filename = hisotry_adapter["filename"]
                if check_adapter_filename_valid(filename):
                    print(f"""Label {label["label_name"]} will use a history one "{filename}" because current one unavailable.""")
                    break
        all_adapters.append(filename)
#adapter_name_in_dir = os.listdir("./save_adapters/")
all_adapter_name = []
with mute_logging():
    for adapter_name in all_adapters:
        name = model.load_adapter(f"{NER_ADAPTERS_PATH}/save_adapters/{adapter_name}")
        all_adapter_name.append(name)
        model.load_head(f"{NER_ADAPTERS_PATH}/save_heads/{adapter_name}")

import re

parallel_text = "','".join(all_adapter_name)
result = re.findall(r'[;|(|)]',parallel_text)
if len(result) != 0:
    raise(ValueError("Adapter Name must not contain \"" + '\", \"'.join(result) + '"'))

from transformers.adapters.composition import Parallel
parallel = eval("Parallel('" + "','".join(all_adapter_name) + "')")

model.set_active_adapters(parallel)

device = "cpu"

def get_adapter_mapping(model):
    print(model.active_head)
    label_2_id_mapping = dict()
    id_2_label_mapping = dict()
    for i, head in enumerate(model.active_head):
        label_2_id_mapping[head] = i
        id_2_label_mapping[i] = head
    return label_2_id_mapping, id_2_label_mapping



def model_predict(model, sentence, device = "cpu"):
    # Check if there is update, if yes, then reload the model.
    tokenized_sentence = torch.tensor([tokenizer.encode(sentence)])
    pos = torch.tensor([[0] * len(tokenized_sentence)])
    tags = torch.tensor([[1] * len(tokenized_sentence)])

    model = model.to(device)
    with torch.no_grad():
        outputs = model(input_ids=tokenized_sentence.to(device), 
                        token_type_ids=pos.to(device), 
                        attention_mask=tags.to(device))

    logits = outputs[1][0]

    return_tags_order = {}
    all_output = None
    for i, output in enumerate(outputs):

        return_tags_order[i] = (model.active_head[i])

        output = outputs[i][0]

        if all_output != None:
            all_output = torch.cat((all_output, output), dim=2)
        else:
            all_output = output
    all_output = torch.sigmoid(all_output)

    output_array = np.array(all_output)
    output_array = output_array.reshape(output_array.shape[-2], output_array.shape[-1])

    label_confidences = []
    for label_confidence in list(output_array):
        label_confidences.append(list(label_confidence.astype(float)))

    #Drop Head and End since it is start/stop Token
    label_confidences = label_confidences[1:-1]

    max_value = np.array(label_confidences).argmax(axis=1)
    trans_func = np.vectorize(lambda x: model.active_head[x])
    out_labels = trans_func(max_value)
    out_labels = list(out_labels)

    out_sentence = tokenizer.tokenize(sentence)

    return out_sentence, out_labels, label_confidences, return_tags_order