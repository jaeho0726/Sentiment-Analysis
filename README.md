# Daily Reflection Sentiment Analysis Project 

## Project Objective 
The purpose of this project is to examine how sentiment score of my daily journal varies with different workload and duty patterns over time.

### Methods Used 
- Text Mining
- Sentiment Analysis

### Technologies / Modules
- Python
    - KoNLPy
    - re
    - numpy
    - pandas

## Project Description
I was taught to gain habit of making a checklist of what I should do tomorrow in military. I initially considered this as one of the remaining absurdities in military. Yet, I later realized this as a important habit I can carry beyond military to record my thoughts and emotions each day. 

Not only limited to daily reflection, I considered this journal as personal text data. Inspired from the text mining concepts learned in data science online course during the service, I decided to conduct a sentiment analysis of my personal journal data and find whether there is a relationship in sentiment score and my workload and duty status. 

## Data Collection
Each daily reflection journal was recorded through **Notion** database with each journal being an entry in the database. Each journal is consist of two numerical data (name, hours of work),  three categorical data (duty status, work intensity, and overall mood), and text data (reflection). 
|Name|Today's Status|Hours of Work|Work Intensity|Overall Mood|Reflection|
|:---|:-------------|:------------|:-------------|:-----------|:---------|
|1/24/2026|On Duty|0|0|9|...|
|2/25/2026|On Duty|2.5|2|6|...|
|...|...|...|...|...|...|
- **Name** : date of daily reflection
- **Today's Status** : represents today's status (military leave / temporary leave / on duty)
- **Hours of Work** : represents how many hours I work in corresponding date
- **Work Intensity** : represents how hard the day was from 0 - 10 with 0 being easy and 10 being intense
- **Overall Mood** : represents how I felt overall from 0 - 10 with 0 being poor and 10 being great
- **Reflection** : text written in Korean about my day 

### Data Export Process
Considering that the data was recorded in Notion, had to export the database into google sheets or Excel in csv form. To simplify the process, I utilized an AI platform **Zapier**, automating exportation of Notion database to **google sheets** when every modification was made in Notion database.  

## Data Pre-Processing
### Data Cleaning
1) **Importing Data from Google Sheets**
   ```
   from google.colab import auth
   auth.authenticate_user()

   import gspread
   from google.auth import default
  
   creds, _ = default()
  
   gc = gspread.Client(auth=creds)
  
   spreadsheet =   gc.open_by_url('https://docs.google.com/spreadsheets/d/1DrDCLitpXZyJajcWYcMNcJCZU4Xa0aE_gj4sfXzl6x4/edit?gid=0#gid=0')
  
   worksheet = spreadsheet.worksheet('data')
   ``` 
2) **Changing the Imported Spreadsheet into Dataframe**
   ```
   raw = worksheet.get_all_records()

   headers = raw[0]
   rows = raw[1:]

   df = pd.DataFrame(rows, columns=headers)
   ```
3) **Changing Column Names & Fixing Data Types**
   ```
   # Shortening Column Names
   df = df.rename(columns={
    "Work Intensity (0 - Easy / 10 - Intense)": "Work Intensity",
    "Overall Mood (0 - Poor / 10 - Great)": "Overall Mood"
   })

   # Fixing Data Types
   df["Hours of Work"] = pd.to_numeric(df["Hours of Work"], errors="coerce").fillna(0)
   df["Work Intensity"] = pd.to_numeric(df["Work Intensity"], errors="coerce").fillna(0)
   df["Overall Mood"] = pd.to_numeric(df["Overall Mood"], errors="coerce").fillna(0)
   ```
   Initially, column names of categorical data included a scale for a reference. Yet, the reference was unnecessary in dataframe.

   Also, the fixed the data type of 'Hours of Work', 'Work Intensity', and 'Overall Mood' to numeric.

4) **Adding Columns for Sentiment Score and Sentiment**
   ```
   sentiment_result =[]

   for i in range(len(df)):
     sentiment_result.append(sentiment_predict_paragraph(df['Reflection'][i]))

   sentiment_score = [item[0] for item in sentiment_result]
   sentiment = [item[1] for item in sentiment_result]

   df['Sentiment Score'] = sentiment_score
   df['sentiment'] = sentiment
   ```

### Data Processing
```
def sentiment_predict_paragraph(paragraph):
  if not isinstance(paragraph, str) or len(paragraph.strip()) == 0:
    return None, "N/A"

  split_result = re.split('[.!?]', paragraph)
  sentences = [s.strip() for s in split_result if len(s.strip()) > 0]

  if len(sentences) == 0:
    return None, "N/A"

  scores = []
  for sentence in sentences:
    try:
      new_token = [word for word in okt.morphs(sentence) if word not in stopwords]
      if len(new_token) == 0:
        continue
      new_sequences = tokenizer.texts_to_sequences([new_token])
      new_pad = pad_sequences(new_sequences, maxlen=max_len)
      score = float(model.predict(new_pad, verbose=0))
      scores.append(score)
    except Exception as e:
      continue

  if len(scores) == 0:
    return None, "N/A"

  avg_score = sum(scores) / len(scores)
  sentiment = "긍정" if avg_score > 0.5 else "부정"
  return round(avg_score * 100, 2), sentiment
```
Above is a function to find out sentiment scores and sentiments of each daily journal. 

1) **Splitting by sentences**
   ```
   split_result = re.split('[.!?]', paragraph)
   sentences = [s.strip() for s in split_result if len(s.strip()) > 0]
   ```

2) **Tokenization & Stopwords Removal**
   ```
   # stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '강', '과', '도', '를', '으로', '자', '에', '와', '한', '하다'] 
   new_token = [word for word in okt.morphs(sentence) if word not in stopwords]
   ```
   

### Model Training
Used pre-existing 


## Results 


## Analysis 


## Limitations 

## Reference
https://www.youtube.com/watch?v=7GUoDHxN5NM

