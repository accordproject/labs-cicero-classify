from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

print("QA Model is loading...")
tokenizer = AutoTokenizer.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad")

model = AutoModelForQuestionAnswering.from_pretrained("bert-large-cased-whole-word-masking-finetuned-squad")


def get_split(text1):
    l_total = []
    l_parcial = []
    if len(text1.split())//150 >0:
        n = len(text1.split())//150
    else: 
        n = 1
    for w in range(n):
        if w == 0:
            l_parcial = text1.split()[:200]
            l_total.append(" ".join(l_parcial))
        else:
            l_parcial = text1.split()[w*150:w*150 + 200]
            l_total.append(" ".join(l_parcial))
    return l_total

def threading_prediction(text, question, poential_answers):
    inputs = tokenizer(question, text_slice, return_tensors='pt')
    start_positions = torch.tensor([1])
    end_positions = torch.tensor([3])

    outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
    #outputs = model(**inputs)
    loss = outputs.loss
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    
    start_index = int(start_scores.argmax())
    end_index = int(end_scores.argmax())

    input_ids = inputs["input_ids"].view(-1)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[start_index:end_index+1])
    answer = tokenizer.convert_tokens_to_string(tokens)
    confidence = int((start_scores.max() + end_scores.max()) / 2)
    poential_answers.append({"answer": answer, "confidence": confidence})

def get_QA_model_answer(question, text):
    text_slices = get_split(text)
    poential_answers = []
    for text_slice in text_slices:
        inputs = tokenizer(question, text_slice, return_tensors='pt')
        start_positions = torch.tensor([1])
        end_positions = torch.tensor([3])

        outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
        #outputs = model(**inputs)
        loss = outputs.loss
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        start_index = int(start_scores.argmax())
        end_index = int(end_scores.argmax())

        input_ids = inputs["input_ids"].view(-1)

        tokens = tokenizer.convert_ids_to_tokens(input_ids[start_index:end_index+1])
        answer = tokenizer.convert_tokens_to_string(tokens)
        confidence = int((start_scores.max() + end_scores.max()) / 2)
        poential_answers.append({"answer": answer, "confidence": confidence})


    poential_answers.sort(key = lambda x: x["confidence"], reverse=True)
    return poential_answers