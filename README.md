# **Daily Reflection Sentiment Analysis Project**

An end-to-end Korean NLP project that transforms daily journal reflections into continuous sentiment scores and evaluates how closely those scores align with independently recorded self-reported mood.

The project compares two sentiment-analysis approaches:

1. KoNLPy/Okt as the baseline
2. KLUE-BERT as the transformer-based approach

Both models are trained on the Korean Naver Sentiment Movie Corpus (NSMC) and applied to personal Korean journal reflections. Their outputs are converted to a 0–10 sentiment scale and evaluated against daily self-reported mood scores.

## **Project Objective**

The goal of this project is not simply to classify text as positive or negative.
Instead, the project investigates whether Korean NLP models can generate a continuous sentiment signal from daily reflections that meaningfully aligns with independently recorded mood data.

The main questions are:

- Can text-based sentiment scores reflect changes in self-reported daily mood?
- How does a traditional Korean NLP + deep learning pipeline compare with a pretrained transformer?
- Can the resulting sentiment signal support further analysis of daily workload, work intensity, and status?

`Overall Mood` is treated as an independent self-reported reference, rather than objective psychological ground truth.

## **Data Collection Pipeline**

## **Project Description**

```
Daily Activity
     |
     v
Notion
     |
     |  Selected entries exported through Zapier
     v
Google Sheets
     |
     |  Python / gspread
     v
Data Preprocessing
     |
     v
Sentiment Analysis
     |
     +---- KoNLPy/Okt + BiLSTM
     |
     +---- KLUE-BERT
     |
     v
Model Validation
     |
     v
Personal Data Analysis
```

Each daily entry contains structured variables together with a Korean-language written reflection.

### **Recorded Variables**

| Variable         | Description                                   |
| :--------------- | :-------------------------------------------- |
| `Date`           | Date of journal entry                         |
| `Today's Status` | Daily work/duty status                        |
| `Hours of Work`  | Number of hours worked                        |
| `Work Intensity` | Self-reported workload intensity from 0 to 10 |
| `Overall Mood`   | Self-reported mood from 0 to 10               |
| `Reflection`     | Daily reflection in Korean                    |
| `Day_of_Week`    | Derived weekday feature                       |

The raw journal dataset is intentionally excluded from the repository because it contains private personal reflections about military life.

## **Repository Structure**

```
Sentiment-Analysis/
|
|-- notebooks/
|   |-- kluebert_baseline.ipynb
|   `-- konlpy_baseline.ipynb
|
|-- src/
|   |-- __init__.py
|   |-- preprocessing.py
|   |-- sentiment.py
|   `-- evaluation.py
|
|-- requirements.txt
`-- README.md
```

`notebooks/`
: contains the end-to-end modeling and analysis workflows including:

- NSMC data loading
- Model training and loading
- Personal journal inference
- Validation
- Exploratory analysis
- Visualization

`src/`
: contrains reusable project logic shared across the notebooks

- `preprocessing.py`
  : handles data cleaning, processing, formatting
  - NSMC text cleaning
  - Journal column formatting
  - Numeric type conversion
  - Date processing
  - Weekday feature generation

- `sentiment.py`
  : handles sentiment inference for both models
  - KLUE-BERT long-text inference
  - KoNLPy/Okt sentence-level inference
  - Token-weighted sentiment aggregation
  - Conversion of sentiment probability to a 0-10 score

- `evaluation.py`
  : provides shared model evaluation logic
  - Mean Absolute Error
  - Root Mean Squared Error
  - Mean Bias
  - Pearson Correlation
  - Largest mood-sentiment discrepancies

## **Methodology**

Two Korean sentiment-analysis approaches were developed and compared.

Both models were trained using the Naver Sentiment Movie Corpus (NSMC) and later applied to the personal daily reflection dataset.

Because NSMC consists of movie reviews while the target data consists of personal journal entries, this introduces an important domain shift that is discussed in the limitations section.

## **1. KoNLPy/Okt Baseline**

The first approach uses traditional Korean NLP preprocessing with a neural network sentiment classifier.

### Text Preprocessing

NSMC reviews are cleaned by:

- Removing duplicate reviews
- Removing missing observations
- Retaining Korean characters and spaces
- Removing empty reviews
- Tokenizing Korean text using Okt
- Removing selected stopwords

The processed sequences are then used to train a Bidirectional LSTM classifier.

### Model Architecture

```
Korean Text
     |
     v
Okt Tokenization
     |
     v
Keras Tokenizer
     |
     v
Embedding Layer
     |
     v
Bidirectional LSTM
     |
     v
Dense Sigmoid Output
     |
     v
Positive Sentiment Probability
```

### Full-Reflection Sentiment Scoring

Daily journal reflections are generally much longer than NSMC movie reviews.

Instead of truncating the entire reflection into a single sequence, each reflection is divided into sentences.

For each sentence:

1. Korean morphological analysis is performed using Okt.
2. Stopwords are removed.
3. The sentence is converted to the trained tokenizer sequence.
4. The BiLSTM produces a positive sentiment probability.
5. Sentence probabilities are combined using token-weighted averaging.

The final probability is converted into a score between 0 and 10:
`Sentiment Score = Weighted Positive Probability x 10`

This method allows longer reflections to contribute more information than a single truncated input.

## **2. KLUE-BERT**

The second approach uses KLUE-BERT, a Korean pretrained transformer model.

The model is fine-tuned for binary sentiment classification using NSMC.

### Why KLUE-BERT?

The KoNLPy baseline depends heavily on explicit tokenization and learned word representations.

KLUE-BERT instead uses contextual representations, allowing the meaning of a token to depend on the surrounding text.

This makes it better suited for sentences where meaning depends on context, phrasing, or combinations of words.

### Long-Reflection Processing

BERT models have a maximum input sequence length.

Rather than truncating journal entries, the full reflection is tokenized and split into smaller token chunks.

```
Full Reflection
     |
     v
KLUE-BERT Tokenization
     |
     v
Token Chunk 1
Token Chunk 2
Token Chunk 3
...
     |
     v
KLUE-BERT
     |
     v
Positive Probability per Chunk
     |
     v
Token-Weighted Average
     |
     v
0-10 Sentiment Score
```

Each chunk receives a positive sentiment probability, and the probabilities are aggregated according to the number of tokens contained in each chunk.

The resulting score is: `Sentiment Score = Weighted Positive Probability x 10`

## **Model Validation**

The main goal of the model comparison is not simply to determine which model performs better on NSMC.

Instead, the project asks '_Which sentiment model produces scores that align more closely with independently recorded daily mood?_'

The generated sentiment scores are therefore compared against the user's self-recorded `Overall Mood` values.

The following metrics are used:

- Mean Absolute Error
  : measures the average absolute difference between sentiment score and self-reported mood
  (Lower values indicate closer alignment)

- Root Mean Square Error
  : measures the difference between sentiment score and self-reported mood with greater weight to large discrepancies

- Mean Bias
  : `Sentiment Score` - `Overall Mood`

- Pearson Correlation
  : measures the strength of the linear relationship between sentiment score and self-reported mood.

## **Model Comparison**

| Model     | MAE ↓     | RMSE ↓    | Mean Bias  | Pearson r ↑ |
| --------- | --------- | --------- | ---------- | ----------- |
| KoNLPy    | 2.029     | 2.401     | -1.518     | 0.196       |
| KLUE-BERT | **1.459** | **1.845** | **-0.597** | **0.482**   |

KLUE-BERT showed stronger alignment with the independently recorded mood scores across all four comparison metrics.

Relative to the KoNLPy baseline:

- MAE decreased by approximately 28%
- RMSE decreased by approximately 23%
- Negative prediction bias was substantially reduced
- Pearson correlation increased from 0.196 to 0.482

The KoNLPy model showed a noticeable tendency to compress sentiment scores toward the lower-middle portion of the scale.

KLUE-BERT produced a wider and more responsive range of sentiment scores and showed a stronger relationship with changes in self-reported mood.

Based on these results, KLUE-BERT was selected as the preferred sentiment representation for downstream analysis.

## **Largest Mood-Sentiment Score Discrepancies**

In addition to aggregate metrics, the project examines observations with the largest differences between:

_Self-Reported Mood vs Model-Generated Sentiment_

These cases are useful because sentiment and mood do not necessarily represent the same concept.

For example, a person may assign a relatively high overall mood score while writing about one specific negative event, or may report a low mood while reflecting positively on certain parts of the day.

Therefore, disagreement between the two measurements is treated as an analytical observation rather than automatically as a model error.

## **Key Findings**

1. **Transformer-based sentiment modeling aligned better with self-reported mood**

   : KLUE-BERT substantially outperformed the KoNLPy/Okt baseline on all mood-alignment metrics, suggesting that contextual language representations are more effective for capturing the emotional tone of longer personal reflections in this dataset.

2. **KoNLPy baseline showed strong downward bias**

   : The baseline model frequently produced sentiment scores substantially below the corresponding self-reported mood score. Its mean bias of -1.518 indicates systematic underestimation relative to the reference mood scores.KLUE-BERT reduced this bias to -0.597.

## **Limitations**

### Domain Shift

Both models are trained using NSMC, which contains Korean movie reviews. The target dataset consists of personal daily reflections.

The language patterns and emotional expressions in these two domains can differ substantially. Future work could use sentiment data that more closely resembles diaries, social writing, or personal reflections.

### Small Personal Dataset

The hardship and adversity of work and daily life differs for each level: _private_, _private first class_, _corporal_, and _sergeant_. The daily reflection sentiment analysis project started when I was a sergeant with only 6 months left until the end of the service. If this project started earlier, it could have allowed a new point of view on analysis: patters in sentiment analysis score for each level. Also, patterns found in this project therefore should not be generalized to broader populations.

## Privacy

The project is built using private personal journal data, which includes military information.

To protect sensitive information:

- Raw journal reflections are not included in this repository.
- Private Google Sheets data is not included.
- Saved models stored in private Google Drive directories are not included.
- The repository focuses on the processing, modeling, evaluation, and analysis pipeline.

This allows the technical workflow to remain publicly accessible without exposing the original journal content.

## **Technologies**

### Data Analysis

- Python
- pandas
- NumPy
- SciPy
- scikit-learn

### Machine Learning / NLP

- PyTorch
- Hugging Face Transformers
- KLUE-BERT
- TensorFlow
- Keras
- Bidirectional LSTM
- KoNLPy
- Okt

### Visualization

- Matplotlib
- Seaborn

### Data Pipeline

- Notion
- Zapier
- Google Sheets
- gspread
- Google Colab

## **Future Improvements**

This project can improve in the future by:

- collecting a longer longitudinal dataset in different settings (non military life)
- exploring temporal mood and sentiment patterns
- investigating mood-sentiment disagreement in greater detail
- comparing additional Korean pretrained language models

## **Takeaway**

This project developed from a simple personal journaling habit into an end-to-end NLP and personal data analysis workflow.

```
Daily Data Collection
        |
        v
Automated Data Pipeline
        |
        v
Structured + Unstructured Data
        |
        v
KoNLPy/Okt + BiLSTM Baseline
        |
        v
KLUE-BERT
        |
        v
Quantitative Model Validation
        |
        v
Personal Data Analysis
```

The project demonstrates how structured daily measurements and unstructured Korean text can be combined into a reproducible machine-learning pipeline.
