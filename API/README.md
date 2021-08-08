# API design

![](https://i.imgur.com/ufZtiK8.png)

Basically, there are 4 job the Classify Model API need to do.

1. Suggest the template to be use by raw text.
2. Predict special KeyWord by raw text.
3. Predict best practice of the model by marked text.
4. Warn User if they make a low-confidence label.

For detail please check [this meeting video](https://vimeo.com/577771919). 

And feel free to suggest that:
1. What API we need.
2. What is the URL of each API
3. What is the format of request and response data?

# Currectly Design:
Document: http://gsoc.demo.eason.tw:13537/docs#/


POST
​/text​/label
Text Label (Mark Keyword)

POST
​/template​/suggest
Suggest Template

POST
/predict/concerto-model
Predict the best practice

GET
​/model​/status
Get Model Status
Optimize


PUT
​/data
Update Data

PUT
​/model​/retrain
Retrain Model
