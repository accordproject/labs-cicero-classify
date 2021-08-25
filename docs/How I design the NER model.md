# How I design the NER model?

<!-- Edit at https://hackmd.io/dT1MhCdhSrmECVDYTFCplg -->

I use [Adapter Transformer](https://docs.adapterhub.ml/quickstart.html#introduction) to train an scaleable NER model.

## How Adapter Work?
![](https://i.imgur.com/aVTwWrY.png)
Adapter first proposed in 2019. It fix the transformers pre-train model's weight, and use a tiny down-stram model, so call Adapter, to interpret the output. When training, it will only do gradient descend on Adapter while keep the Pre-Train model weight fix. The paper had shown that, by using Adapter, the performance is almost even as the conventional methods which train the whole model. Yet, the training time for the former is far less than the latter.

Furthermore, we can use [Parallel](https://docs.adapterhub.ml/adapter_composition.html?highlight=stack#parallel) calculation for multiple Adapters. That is, we can use many Adapters on prediction, and Adapters won't affect each other. For more detail, Please reference [the paper](https://arxiv.org/pdf/1902.00751.pdf).


![](https://i.imgur.com/TCSGnjl.png)

In the picture above, each Adapter represents a label in the NER model. And All of them are predicting parallelly using the same output from the BERT pre-train model, which weight keeps freezing.


![](https://i.imgur.com/SJLoBrZ.png)

When training, we will only do the training step (Gradient Descent) on the specific Adapter.

## FYI:
- [How Backend Train and Use NER Adapters?](https://github.com/just_not_add_it_yet)
