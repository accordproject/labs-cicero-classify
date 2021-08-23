# How to process the labeled data?
> This part show how to put data into mongoDB and read it to be train.

I process the data into tokenized CONLL format.

Input Sentence: *Eason will meet Dan at 2021-08-04 18:00.*

After split: ['Eason', 'will', 'meet', 'Dan', 'at', '2021-08-04', '18:00.', ] 

And there also have tags: *[['Party', 'String'], ['O'], ['O'], ['Party', 'String'], ['O'], ['TemporalUnit'], ['O']]*

Front-end need to process data and put it onto JSON "texts" attribute before call the API `POST /data/label` to add label into dataset:

```
{
  "user": "example@gmail.com",
  "texts": [
    {
      "text": "Eason",
      "labels": [
        "Party",
        "String"
      ]
    },
    {
      "text": "will",
      "labels": [
        "O"
      ]
    },
    {
      "text": "meet",
      "labels": [
        "O"
      ]
    },
    {
      "text": "Dan",
      "labels": [
        "Party",
        "String"
      ]
    },
    {
      "text": "at",
      "labels": [
        "O"
      ]
    },
    {
      "text": "2021-08-04 18:00",
      "labels": [
        "TemporalUnit"
      ]
    },
    {
      "text": ".",
      "labels": [
        "O"
      ]
    }
  ]
}
```

When received, backend will transform it into the CONLL format and store in DB, The CONLL format is the following:

| Sentence # | text | labels |
| -------- | -------- | -------- |
1 | Eason | ['Party', 'String']
1 | will | ['O']
1 | meet | ['O']
1 | Dan | ['Party', 'String']
1 | at | ['O']
1 | 2021-08-04 18:00 | ['TemporalUnit']
1 | . | ['O']


And because I use RoBERTa as the model, and run the tokenizer is cost a lot of time, so I will also store pre-tokenized CONLL format in DBs' token_and_labels column.

Tokenizer will transform text from 

['Eason', 'will', 'meet', 'Dan', 'at', '2021-08-04', '18:00.', ] 

to 

['E', 'ason', 'Ġwill', 'Ġmeet', 'ĠDan', 'Ġat', 'Ġ2021', '-', '08', '-', '04', 'Ġ18', ':', '00', '.', ]

Note that `Ġ` in RoBERTa Tokenizer means that there is a white space in front of the token. For example, "Eason" will be encoded as ['E', 'ason'], but "E ason" will be encoded as ['E', 'Ġason'] ([Check this answer from Slack Overflow for more detailed and example](https://stackoverflow.com/questions/62422590/do-i-need-to-pre-tokenize-the-text-first-before-using-huggingfaces-robertatoken))

| Sentence # | text | labels |
| -------- | -------- | -------- |
1 | E | ['Party', 'String']
1 | ason | ['Party', 'String']
1 | Ġwill | ['O']
1 | Ġmeet | ['O']
1 | ĠDan | ['Party', 'String']
1 | Ġat | ['O']
1 | Ġ2021 | ['TemporalUnit']
1 | - | ['TemporalUnit']
1 | 08 | ['TemporalUnit']
1 | - | ['TemporalUnit']
1 | 04 | ['TemporalUnit']
1 | Ġ18 | ['TemporalUnit']
1 | : | ['TemporalUnit']
1 | 00 | ['TemporalUnit']
1 | . | ['O']


That is. We are ready to put our data onto the NER model now.


## Appendex A: Cache
And the NER_trainer have a [cache macanism](https://github.com/accordproject/labs-cicero-classify/blob/dev/API/utils/trainer/NER.py#L72), That is, store cache data in CSV instead of reading from MongoDB. To save time.

Since CSV can't store List object, when store the data, `labels` will be transform to `cache_labels` by `lambda x: "|".join(x)`. And `cache_labels` will be load by `lambda x: x.split("|")`. Just Like the following example:

| Sentence # | text | cache_labels |
| -------- | -------- | -------- |
61 | Eason | Party\|String
61 | will | O
61 | meet | O
61 | Dan | Party\|String
61 | at | O
61 | 2021-08-04 18:00 | TemporalUnit
61 | . | O