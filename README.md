# **Daily Reflection Sentiment Analysis Project**

## **Project Objective** 
The purpose of this project is to examine how the sentiment score of my daily journal varies with different workload and duty patterns over time. By leveraging Natural Language Processing (NLP), I aim to quantify subjective meotional data to find correlations with objective work metrics.

### **Methods & Technologies**
- Language: Python

- Core ML Frameworks: PyTorch, Hugging Face Transformers (**KLUE-BERT**)

- Legacy Frameworks: Keras, Tensorflow, **KoNLPY (Okt)**

- Data Engineering: Zapier (Data Transfer Automation), Google Sheets API, Pandas, NumPy


## **Project Description**
In the era led by Artificial Intelligence (AI), I believe that the irreplaceable, unique value of humanity lies in 'emotions.' And I am convinced that the medium which best projects this unique human value is 'language.' I consider that keeping a diary, where we express and record our emotions through our own languages, is not merely a form of writing, but rather essential 'data' in an AI-centric era. 

Developing on top of the experience from military service, which I developed a habit of recording my thoughts and emotions daily, I decided to treat my own daily journal as a personal dataset. Utilizing text mining concepts from data science course I took during the service and data science skills learned in my freshman year at UCSD, I look forward to determine if a measurable relationship exists between my daily journal sentiment scores, work intensity, and duty status. 



## **Data Collection**
Each daily reflction journal was recorded through **Notion** database. Each journal is consist of two numerical data _(name, hours of work)_,  three categorical data _(duty status, work intensity, and overall mood)_, and text data _(reflection)_. 
|Name|Today's Status|Hours of Work|Work Intensity|Overall Mood|Reflection|Exportation|
|:---|:-------------|:------------|:-------------|:-----------|:---------|:----------| 
|1/24/2026|On Duty|0|0|9|...|True|
|2/25/2026|On Duty|2.5|2|6|...|True|
|...|...|...|...|...|...|...|
- **Name** : date of daily reflection
- **Today's Status** : represents today's status (military leave / temporary leave / on duty)
- **Hours of Work** : represents how many hours I work in corresponding date
- **Work Intensity** : represents how hard the day was from 0 - 10 with 0 being easy and 10 being intense
- **Overall Mood** : represents how I felt overall from 0 - 10 with 0 being poor and 10 being great
- **Reflection** : text written in Korean about my day
- **Exportaion** : a checkbox whether the daily reflection can be exported from Notion to Google Sheets

### **Data Exportation**
The data recorded in **Notion** database is automatically exported to **Google Sheets** via **Zapier**. The automatic exportation is made when the value of 'Exportation' property is true in Notion database. 

## **The Evolution of Model Architecture**
### Phase 1: LSTM with KoNLPy (Initial Baseline)
Originally, I built a **LSTM** using Keras. This model utilized the **Okt** morphological analyzer from the **KoNLPy** library for tokenization. While this provided a functional baseline, it faced challenges:

- **Static Embeddings**: it struggled with the nuanced, contextual nature of personal reflections

- **Preprocessing Complexity**: required manual stopward removal and sequence padding

### Phase 2: KLUE-BERT (current model)
To improve performance, I migrated the pipeline to **KLUE-BERT** (Korean Language Understanding Evaluation - BERT)

- **Why BERT?**: unlike LSTM that processes text in one direction, BERT's bidirectional attention mechanism understands the context of a word based on its surroundings

- **Transfer Learning**: KLUE-BERT is pre-trained on massive Korean corpora, allowing it to perform accurately even wit my relatively small personal dataset

- **Subword Tokenization**: replaced 'Okt' with a subword tokenizer, effectively handling 'Out-of-Vocabulary' words and reducing the need for manual stopword filtering

## **Technical Implementation (KLUE-BERT)**

### Model Validation: Ground Truth Correlation
To ensure the model's sentiment score was accurate, I performed a **Pearson Correlation** analysis against my manually recorded 'Overall Mood'. This validation step ensures that the mode's view of my day aligns with my evaluation of my mood. 

### Sentiment Polarity Transformation
To make the data more interpretable for visualization, I transformed the Softmax probability output [0,1] to a **Centered Polarity Score** [-1,1]. 

$$Score_{Polarity} = (Score_{Softmax} - 0.5) \times 2$$

- -1 : Strong Negative Sentiment
- 0 : Neutral
- 1 : Strong Positive Sentiment

## **Results & Analysis** 
... 


## **Limitations & Challanges**
- **Hierarchical Constraints**: The hardship and adversity of work and daily life differs for each level: _private_, _private first class_, _corporal_, and _sergeant_. The daily reflection sentiment analysis project started when I was a sergeant with only 6 months left until the end of the service. If this project started earlier, it could have allowed a new point of view on analysis: patters in sentiment analysis score for each level. 

- **Domain Gap**: The model was fin-tuned on the **Naver Movie Sentiment Corpus (NSMC)**. While being powerful, movie reviews and personal journals have different linguistic structures, which I addressed by utilizing the contextual strengths of BERT. 


## **References**
- [KLUE-BERT](https://www.youtube.com/watch?v=7GUoDHxN5NM)

- [NSMC Dataset](https://github.com/e9t/nsmc)

- https://www.youtube.com/watch?v=7GUoDHxN5NM

