import pandas as pd
import numpy as np
import torch
print(f"Torch Version: {torch.__version__}")

import transformers
print(f"transformers (Adapter) Version: {transformers.__version__}")

from transformers import RobertaTokenizer
import numpy as np

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")

data_path = "../Practice/adapter_roberta/NER_multilabel_data_v2.csv"
df = pd.read_csv(data_path)

all_tags = df.newTag

all_tags = set(all_tags)

all_tags = "|".join(all_tags)
all_tags = all_tags.split("|")
all_tags = set(all_tags)
all_tags = list(all_tags)

from ner_dataset import get_trainset_data_loader

all_tags, trainset, trainloader = get_trainset_data_loader(tokenizer, BATCH_SIZE=128)


from transformers import RobertaConfig, RobertaModelWithHeads

config = RobertaConfig.from_pretrained(
    "roberta-base",
    num_labels=len(all_tags),
    label2id = trainset.label_map, 
    id2label = trainset.id2label
)
model = RobertaModelWithHeads.from_pretrained(
    "roberta-base",
    config=config,
)

all_adapter_name = []
for tag in all_tags:
    adapter_name = f"{tag}_0731"
    name = model.load_adapter(f"../Practice/adapter_roberta/save_adapters/{adapter_name}")
    all_adapter_name.append(name)
    model.load_head(f"../Practice/adapter_roberta/save_heads/{adapter_name}")

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