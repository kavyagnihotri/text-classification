# Natural Language Processing 
> Deadline: 20th June, Friday 13:00 o'clock CET

## Task: Text Classification
### Steps:

#### 1. choose task

#### 2. choose dataset from HuggingFace
You may explore the list of available datasets by running the following commands in a Google colab notebook:
``` 
!pip install datasets
from datasets import list_datasets
datasets_list = list_datasets()
print(datasets_list)
```
It is important to use one of the listed datasets by directly importing them from the above library.

#### 3. Carry out the following steps:

1. Read the dataset, clean and preprocess as required
2. Report variouos statistics that offer a good understanding of the dataset. Eg. document count,label count, token count, unigram count, bigram count **and more** as discussed in the preprocessing lecture
3. Explain each step and your reason behind them
4. Implement **two** exisitng models/feature settings for the task, evaluate and compare the models based on **all** suitable evaluation metrics. *Note: at least one model needs to be **Neutal network based model***
5. Discuss your observations from the comparison and jusity the reasons behind the results

> Explore your favourite sets as input to the models


#### 4. Report
Write a report and submit together with your colab notebook, report in PDF format.


## Grading Overview:


