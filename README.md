# labs_cicero_classify
This project investigates the automatic Identification And Classification of Contract Data Types with NLP Models

## Week 1 (06/07 - 06/13)

This week I mainly prepare my final exam (end at 6/11).

### 6/7
I have a discuession with my mentor, Niall, for kick-off and the plan on the following week.

Niall guide me walk through templates patiently to help me understand the relationship between Templates, DataType, and Models. ([Meeting Notes](https://hackmd.io/@EasonC/Bk6rH_i5_))

### 6/12

#### Start with good tools
I set up the environments for this project, include this GitHub-Repo and the manage trello, miro board.

#### Learn on reading documents.

I read ERGO and CTO document to help me understand more about them. 

#### Spider templates
I ["spider" all templates](https://github.com/EasonC13/GSoC_Accord_Project/blob/main/crawler_data/0612_crawler_files.ipynb) form [Template Studio](https://templates.accordproject.org/) via python's file walkthrough the [File from Git Repo](https://github.com/accordproject/cicero-template-library).

This job have two propose:
1. Data mining on template studio to help me understand the data.
2. Prepare data for model training step afterward.

#### Data mining on template studio's data
I [count keywords](https://github.com/EasonC13/GSoC_Accord_Project/blob/main/crawler_data/count_keyword.ipynb) (ex: buyer, seller, value) and [count datatype](https://github.com/EasonC13/GSoC_Accord_Project/blob/main/crawler_data/count_object.ipynb) (ex: Party, String) to understand the amount of data avalialbe and the summary of them.

I also found a bug on template library, so I [open an issue](https://github.com/accordproject/cicero-template-library/issues/393) about it.

#### Discuess with my NLP professor.
I discuess with my NLP professor Sam for hierarchical clustering job. We figure out that our goal might be able to done via the BERT NER model with some modification on the last layer. (change to softmax output)

### 6/13

#### Create the PPT for prototype on how user interact with model on template studio.
I create the PPT for prototype on how user interact with model on template studio. I think it could work like [Grammarly](https://app.grammarly.com/). That is, user paste their contract text, then model show the prediction of the keyword and datatype, finally, user correct them and export the contract file.

I will use it to discuess with Niall at Monday. Hope it can be demo on Wednesday's Technology-WG meeting.

#### Plan on next week
I think [these amount of data](https://github.com/EasonC13/GSoC_Accord_Project/blob/main/README.md#data-mining-on-template-studios-data) is worthy for a try. (via pre-train model and snorkel)
So my next step will devide into four part:

1. create Pipeline to Prepare data for try-run NER model training.
2. Learn on how to use Snorkel to augmented data.
3. Research and implement on edit NER model let it have softmax-like output.
4. Discuss UX/UI for how end user to use model on template studio with mentor.


## Week 2 (06/14 - 06/20)

### 6/14
#### Read Document
I read through the [Model-Concerto Document](https://docs.accordproject.org/docs/model-concerto.html) detailly. The concerto-model will be the predicted dataType labeled of the NLP model.

And I have a nice discussion with Niall. [(Meeting Notes)](https://hackmd.io/SyGmgFSGRo-VLcqRnJL4YQ)

### 6/15

#### ProtoType Demo Video
I film a demo video show the UI Prototype which user can interact with the NLP model at Template Studio.
Because by doing so, user can correct the model’s output before export the contract easily, and public can contribute to label data.

Video Link: https://youtu.be/PARJ2VnCpXc

<img src="https://i.imgur.com/NzpCHwS.png" width="300">

<img src="https://i.imgur.com/RrRpsSO.png" width="300">



#### Planning on the next step

Also, I am planning on the next step.

I think ["Research and implement on edit NER model let it have softmax-like output"](https://github.com/EasonC13/GSoC_Accord_Project/blob/main/README.md#plan-on-next-week) will be the first priority. Because we can only know the data-type after knowing the model's workflow. And then we can augmented data base on that format.

I will start try using simpleTransformer to do it.

### 6/17

#### Demo Prototype to Working Group
I join the technology-WG meeting at 04:00 am UTC+8 (Taiwan's time)

I demo the video to the working groups, everyone were feeling exciting about it! And Michael has join the channel, want to discuss UX/UI with me.

### 6/18

#### Get GPU resources.

So far, I have get permission to use me, my internship company’s and my lab’s GPU. So now I have 1 3090, 1 2080ti and 1 Titan RTX available. I hope this is enough for me on try and error the model.

Today, I set up the CUDA ready environment on them.

Now I’m looking on BERT NER model. try to split the last layer to let it have softmax output.

### 6/19 - 6/20

#### Edit NER Model's Last layer to have custom multi labels
I [try simpletransformer](https://github.com/EasonC13/GSoC_Accord_Project/blob/main/Practice/SimpleTransformer/0618_tutorial_1.ipynb), while it is too high-end. I might need to go down to PyTorch Level to edit the model. 

So I looking on the PyTorch and s[uccessfully custom the last layer](https://github.com/EasonC13/GSoC_Accord_Project/blob/main/Practice/PyTorch/NER_test/try_decompose.ipynb). Next step is preparing data and training pipeline from existing NER dataset with custom multi label to set up a pipeline to fine-tune the model.

If this work, I will start to label the data I get from contract templates.

#### Plan on next week

1. Presentation 10 min on Technology WG meeting at 6/24 (UTC+8).
2. preparing data and training pipeline from existing NER dataset with custom multi label output, then set up a pipeline to fine-tune the model.
3. Discuss to find out the label scructure we want model to predict.
4. Learn Snorkel

