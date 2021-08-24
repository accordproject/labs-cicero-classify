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

## Week 3 (06/21 - 06/27)

### 6/21


#### Custom Model
I keep learning to use PyTorch to decompose the model. I want the NER model behave as the picture shown below.

<img src="https://i.imgur.com/WzxFujn.png" height="300">


#### Meeting
I have the weekly meeting with my Mentor, Niall. And Walter join the meeting, too, who discuss NER model with us. ([Meeting Note](https://hackmd.io/@EasonC/HyHlZOpou))

### 6/22 - 6/23

#### Prepare presentation

I prepare the presentation on WG meeting at 6/24 (UTC+8).

### 6/24

#### Accord Project Tech Working Group Meeting

Meeting Recording (My part start after 41:15): https://vimeo.com/566926468

Eventhough my network is out at 31:00, I luckly pre-record the presentation video to prevent me from oversleep. So I give Jerome the video link then use the time playing video to set-up 4G connection to answer questions after video.

### 6/22 - 6/27

Keep going on research.
#### Keep Looking on Custom Model

at 6/24, I think I am encounter a challenge.
It is easy to edit BERT NER model to have custom label. So the model can predict with one label, like “Eason” is a person.
However, I still looking on how to let NER model have “multi custom label”, I want the model to know, like, “Eason” is not only a person but also a party and a string. Because the NER example I see so far, BERT turn the label into a 128 dim label ids, not like classic classification model is the one-hot encoding.
Even-though the model at the pytorch print said it is a n dim output, the actual input data seems not like this.

So I ask my mentors for help, and turn to Keras because I am more familiar with it. ([link](https://github.com/accordproject/labs-cicero-classify/issues/2))

#### Reach with Keras 6/25
Then, 6/25, I successfully change the model output as I wish to categorial mode via Keras.
https://github.com/accordproject/labs-cicero-classify/blob/dev/Practice/keras/keras_decompose_NER_model.ipynb

I first reference this tutorial.
https://apoorvnandan.github.io/2020/08/02/bert-ner/

Then I change the classical NER model's loss to CategoricalCrossentropy and transform train_y into the one-hot encoding format.

Now the model is training! I can't wait to validate it and create a multi_label dataset to verify this method will work or not.

I think I can do it since I first learn Deep learning via Keras. Now I need to know how to do the same thing via PyTorch. I will start reading the document and asking others on it.

#### Keep learning Pytorch.

At 6/25 - 6/27, I keep learning Pytorch. While reading the torials, I also learn more about transformers and python functions. 


### 6/27

#### Data labeling
I use Miro to decide the hierarchical clustering structure of Data.

<img src="https://i.imgur.com/FVMcHBH.png" height="300">

This will be discuss on next weekly meeting with my mentors.

#### GitHub Issue Board usage:

I and my mentors agree that we can use the [issue board](https://github.com/accordproject/labs-cicero-classify/issues) on [labs-cicero-classify](https://github.com/accordproject/labs-cicero-classify) to discuss since Walter can't join Slack.


#### Plan on next week

- Keep looking on how to build custom model via PyTorch
- Create the data labeling pipline.
- Learn Snorkel since it will help data labeling.


## Week 4 (06/28 - 07/04)

This week I build the custom model and training pipeline in PyTorch. The model's performance and prediction is impressive.

### 6/28
#### Meeting WIth Niall and Walter

I have a meeting with my mentor, Niall, and his colleague Walter. 

We define the higherical relation ship between data-type label.

<img src="https://i.imgur.com/gaUeQb5.png" height="300">

Then check the [Adaptor](https://docs.adapterhub.ml/training.html) for transfer learning.

[Meeting Notes](https://hackmd.io/SJptyfORRNOGHbgMPl6hGA)


### 6/29
#### Define the multi label NER dataset format
I define the multi label NER dataset format like the following format, the multiTags will split by "|".



| Sentence | POS | multiTags |
| -------- | -------- | -------- |
| Eason     | -     | B-per\|Party\|String     |
| will     | -     | O     |
| push     | -     | O     |
| his     | -     | O     |
| GSoC     | -     | B-eve\|Event\|String     |
| final     | -     | O     |
| Code     | -     | Object\|String     |
| to     | -     | O     |
| GitHub     | -     | I-org\|Party\|String |
| before     | -     | O     |
| 2021-08-17     | -     | B-tim\|TemporalUnit\|String     |
| .     | -     | O     |



#### Build the dataloader of "multi hot" encoding labels.
I build the dataloader to transfer the multi label NER dataset into "multi hot" encoding labels.

### 6/30 - 7/4
#### Use PyTorch to build and train the custom model
I build up a training pipeline and the multi-label custom model with BCE Loss. Therefore, the model can consider each label as a confidence rate between 0 and 1 respectively. 


([check the full code from dataloader to model train and evaluate here](https://github.com/accordproject/labs-cicero-classify/blob/dev/Practice/PyTorch/BERT_practice/BERT_BASE_Custom_model_multi_label.ipynb)).

#### [Result Demo Video (Click to go to YouTube to view)](https://youtu.be/jkUUPHh553Y)

<a href="#result-demo-video-click-to-go-to-youtube-to-view" target="_blank"><img src="https://user-images.githubusercontent.com/43432631/124401284-0fb1b900-dd5b-11eb-8b61-0cc6e4e5ed25.png" width="600" ></a>

<a href="#result-demo-video-click-to-go-to-youtube-to-view" target="_blank"><img src="https://user-images.githubusercontent.com/43432631/124400310-040ec400-dd54-11eb-9624-06749eab8564.png" width="600" ></a>
#### Plan on next week

- Learn Snorkel since it will help data labeling and augmented the data.
- Learn Adaptor for easy transfer learning and better performance.
- Host an UX/UI Meeting with Niall, Michael, and Matt to discuss the interface where user use the model.


## Week 4 (07/05 - 07/11)

This week I mainly explore the adapter to find how it can help me building the model. And set up the first UX/UI Meeting with Niall, Michael and Matt.

### 7/5 

#### Meeting With Niall, Walter, Saajit, Jiangbo, Madeleine

Niall introduce other to me since they are also working on NER and contract NLP.

[Meeting Notes](https://hackmd.io/QjXlR-djQx2IDeJ9u56cbQ)

### 7/5 - 7/10 
#### Learning Adapter

I think I can use N adapter for N labels, each adapter specialist for one datatype. So when user correct, it is easy to re-train them without interrupt others.

Now I'm trying to implement it.

While it seems adapter only can train with same output label. So it is a little challenge to set up an effective training pipeline, or I should train N times for N adapter. Now I have come up with a plan to done effective training. That is, train all wanted adapter together with one label (ex: 1), no matter it is the correct label or not, but only do gradient descent on those with correct label, then train the opposite label (ex: 0) and do gradient descent on others.

While I first will build a prototype which have N adapter for N labels, on basic NER dataset, to prove adapter will work for this job.

### 7/7
#### Download and Explore Contract text raw data
Niall have provide me some contract text rawdata on [this repo](https://github.com/TheAtticusProject/cuad), which can be use on the weak labeling afterward.

### 7/9 
#### UX/UI meeting with Niall, Michael and Matt.

Result is that, for user to use the model, We will first build an standalone one, focus on the prototype and POC, which can generate an JSON format. Then intergrated it to template editor afterward. the UI component can be re-used.

We set up a user story with Michael and Matt. Then I will look into the model and try to generate the JSON format of the left hand side of the prototype. While Michael, an UX/UI designer, will adapt contract editor design-mode mockups to include classifier.

[UX/UI Meeting Note](https://hackmd.io/bb0az25GSa2vOmivMkADTw)

<img src="https://i.imgur.com/Am8rilp.png" height="300">

### 7/9 
#### Discuss Snorkel and Adapter with Walter

This meeting actually bring everone come, with Niall, Walter, Saajit, Jiangbo, Madeleine.

Walter teach us how snorkel and adapter work, and how they can intergrated in our project.

[Meeting Note](https://hackmd.io/u0BwLm95T4ak9PIyrVTihQ)

### Plan on next week

- Build and train an prototype which have N adapter for N labels, on basic NER dataset, to prove adapter will work for this job.
- Prepare for next UX/UI Meeting
- Learn Snorkel and try to build one to recognize Party from contract dataset.


## Week 5 (07/12 - 07/18)

This week, I mainly focus on prepare the presentation on COSCUP and TAICHI, relative little progress on GSoC.

The First Evaluation has finish at 7/14, and I pass! Hope I can keep going.

### 7/12 
#### Meeting With Niall

I have a Quick meeting with my mentor, Niall. [(Meeting Notes)](https://hackmd.io/XZluEA5PQqmc3lwtvXY1Gg)

### 7/17 

#### Manual Label Functions
I work on Manual Label Functions, which point out dataType like Integer, CountryCode... etc [(code here)](https://github.com/accordproject/labs-cicero-classify/tree/dev/Practice/PyTorch/NER_test/label_functions). Also please [check the README about functions](https://github.com/accordproject/labs-cicero-classify/tree/dev/Practice/PyTorch/NER_test/label_functions#label-functions).

I will use these function on Snorkel later.

#### Inspiration from WG meeting

I check the [WG meeting video](https://vimeo.com/575395928), Matt have build an awesome model builder which allow user to define and create new model. Moreover, Jerome show how to transfrom between `.cto` and JSON format.

Therefore, I got a vision, that My model's output should look like Jerome's JSON format.

So it seems can be like [Scratch](https://scratch.mit.edu/), user can drag and drop the Attribute of each Model and Object. What the NLP model do, is recognize the poential attribute so the user can choose them easily.

Furthermore, User also can select "one click complete" function, model will give a fit on those attribute. then User manually correct them later.

After the label is finish, it can be export to the JSON format as Jerome shown.

Next week I will build some prototype base on it.

### Plan on next week

- Prepare WG meeting presentation.
- Prepare for next UX/UI Meeting.
- Learn Snorkel and try to build one to recognize Party from contract dataset.
- Build and train an prototype which have N adapter for N labels, on basic NER dataset, to prove adapter will work for this job.


## Week 6 (07/19 - 07/25)

### 07/19
#### Meeting with Niall

I have a meeting with Niall and Walter, we talk about the WG Presentation plan, then talk about how to label some datatype like Timezone or Address. Walter suggest us can use some [Timezone converter](https://howchoo.com/g/ywi5m2vkodk/working-with-datetime-objects-and-timezones-in-python) to align different timezone to a general format. And we also think we can use [DateParser](https://dateparser.readthedocs.io/en/latest/) to do it. Both will take into account. ([Meeting Note](https://hackmd.io/h91bW8_3RdecaCGKvLEkVA?edit))

### 07/21
#### UX/UI Meeting with Niall, Michael and Jerome

Me, Niall, Michael and Jerome talk about how user will interact the NLP model to create concerto contract model. Then brif discuss the poential user persona. We think we will arrange another meeting with poential user to get detail of user persona.
[Meeting Note](https://hackmd.io/pmH2HG5iQq6MoUB6wSdpDg)

### 07/22
#### WG meeting presentation.

I have a monthly presentation at Working Group meeting. [Meeting Record Link](https://vimeo.com/577771919). WG like the Idea that intergrated existing component with the model userflow. 

Dan will hold an follow up meeting with me and Niall about defining model's API with Accord Project's existing service.

Also, there are a poential user, Parsa Pezeshki are interested on useing the model and reach out at slack channel.

Then, I took my second dose of vaccine, so have a two day break.

### 7/25
#### Prepare data to recognize Party from contract dataset

At Sunday, I prepare data for multiple label NER dataset, plan to build a adapter base NER model next week. [(Code)](https://github.com/accordproject/labs-cicero-classify/blob/dev/Practice/PyTorch/NER_test/0725_label_dataset_v2.ipynb)

## Plan on next week
- Build and train an prototype which have N adapter for N labels, on basic NER dataset, to prove adapter will work for this job.
- Arrange meeting for user persona.
- Arrange meeting for API connect with Dan

## Week 7 (07/26 - 08/01)


### 07/26
#### Meeting with Niall

I have a meeting with Niall, Niall is going to a vacation next two week. Hope we can meet Dan before Saturday. [Meeting Notes](https://hackmd.io/t_pn8YGUSuOVaWMQ-zstBg)

Moreover, I will start writing my commit message [in more formal way](https://github.com/accordproject/techdocs/blob/master/DEVELOPERS.md#commits). 

### 07/27
#### Set up environment for convenience git use
I set up the CUDA environment, then put my code from docker container to outside system, so I can use VScode to edit the git via ssh remotely. That is far more easy than commend line.

### 07/28 - 7/31
#### Create Multi Adapter for Each NER Label
I write the training pipeline to train multi Adapter septerately for each tag [from dataset I create before](https://github.com/accordproject/labs-cicero-classify/tree/dev#prepare-data-to-recognize-party-from-contract-dataset). Just like I said at the [WG meeting](https://github.com/accordproject/labs-cicero-classify/tree/dev#wg-meeting-presentation). After lots of try and error, it work now finally. Please Reference the code [here](https://github.com/accordproject/labs-cicero-classify/blob/dev/Practice/adapter_roberta/Adapter_train_one_by_one_device0.ipynb).

And the performance of this approach is almost same as the original NER model. [Please check the code here for performance test](https://github.com/accordproject/labs-cicero-classify/blob/dev/Practice/adapter_roberta/load_multi_adapter.ipynb). Moreover, the training time of adapter is super-fast! within 10 min I can finish one tag's fine-tune.

The most important thing to note from my try, is that when add new Adapter, model need to send to GPU device again. Then need to re-setup the optimizer from the new model. Moreover, I [can't train multiple adapter at once since GPU will out of memory](https://github.com/accordproject/labs-cicero-classify/blob/dev/Practice/adapter_roberta/fail_Adapter_custom_model_train_together_0730.ipynb) (Still don't know why).

The Adapters include: ['Float','TemporalUnit','I-gpe','CountryCode','CurrencyCode','Timezone','CryptoCurrencyCode','Month','Party','B-tim','I-art','Time','B-per','B-gpe','B-geo','O','Location','Event','I-nat','Race','B-org','I-geo','I-tim','I-eve','SpecialTerm','B-art','US_States','B-eve','I-org','B-nat','Object','I-per','Integer']

##### Float's Problem
However, I notice that BERT or RoBERTa can't recognize Float, because it will be unrecognizeable at Tokenizer's text to ids parsing, and when predict, Float will be seperate it to pieces then model will think it is integer.

For example, "13.13" in RoBERTa Tokenizer's token to id will be [100] mean unknown. and when predict from raw text, it will be ["13", ".", "13"] then put into model in three words. Therefore, I might need to re-train Float's Adapter by other well-designed float generator dataset that can help model recognize Float.

##### Augmented data with Adapter
And the good news is, now the labels are able to train seperately. So I can finally use Name-Generator to let Party's adapter recognize more name without affect other label. And it also can be work on TimeZone, Currency code, Country Code's detection.

### 7/31
#### API Mockup
Dan, Jerome and Matt have create a private Chat with me to discuss API connect with my model. Now I'm writing the API Mockup [here](https://github.com/accordproject/labs-cicero-classify/tree/dev/API). And I will linked it to NER model afterward. Hope we can have a meeting soon with the API connect.

## Plan on next week
- Augmented data with Adapter
- API Mockup
- Prepare meeting for API mockup

## Week 7 (08/02 - 08/08)

This week my progress is relative little since I spend time on other stuff. (Feeling ill these day). Will catch up at to following week.

#### API Mockup Progress

I change the [API Example Mockup](https://github.com/accordproject/labs-cicero-classify/blob/dev/API/example_mockup.ipynb) for more RESTful.

Preview at https://gsoc.demo.eason.tw/docs

There still have many discussion about it. I will keep iterate the API afterward. And now I will keep focus on the Model part.

#### Plan on Tokenizer-fit dataset

I plan to re-construct the dataset to let it fit RoBERTa's Tokenizer.

### 08/05
#### WG Meeting talk

According to previous meeting, The ML model will do four tasks:
1. Suggest the template to be use by raw text.
2. Predict special KeyWord by raw text.
3. Predict best practice of the concerto model by marked text.
4. Warn User if they make a low-confidence label.

I have present it at WG meeting and looking for feedback about other need of the ML model.
## Week 8 (08/09 - 08/15)

### 0809
#### Meeting with Walter

I have a meeting with Walter and Jiangbo. 

##### Snorkel
We discuss how Snorkel will benefit my project. Walter show me how to let snorkel label NER dataset, and also demonstrate a fluent use of PyCharm, that is super cool! While I think I will first work on the model part, snorkel's data augmentation and multi labeling function can wait until after I have multi label function.


##### Tokenizer
And we also discuss about how to let model know the Tokenizer, Jiangbo agree that I can use RoBERTa's Tokenizer to re-construct the NER dataset to let it fit the model. Will write the pipeline afterward.


##### QA for best practice fit
and we How to let the model "Predict best practice of the concerto model by marked text". Jiangbo and Walter suggest that we can use QA model to do it. 

That is, provide the context and JSON file, then ask model where is the best fit of each JSON attribute (ex: where is Buyer's Address). 

![](https://i.imgur.com/8m71caN.png)

And then model will also can be iterate by user provide more detail information. Like if user mark Eason is a Buyer, the Question will change the question to more specificly to "Where is Eason's Address?"

![](https://i.imgur.com/cvsT1ET.png)

