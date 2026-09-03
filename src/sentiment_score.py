import re

import numpy as np
import torch
import torch.nn.functional as F

from tensorflow.keras.preprocessing.sequence import pad_sequences


def predict_bert_sentiment(
    text,
    model,
    tokenizer,
    device,
    max_length=128
):
    """
    Generate a 0-10 sentiment score for a full reflection
    using KLUE-BERT.

    Long reflections are divided into token chunks.
    Chunk probabilities are aggregated using token-weighted averaging.

    Parameters
    ----------
    text : Full daily reflection. (str)

    model
        Fine-tuned KLUE-BERT sentiment model.

    tokenizer
        Tokenizer associated with the BERT model.

    device
        Torch device used for inference.

    max_length : Maximum token sequence length used by the model. (int, default=128)

    Returns
    -------
    Sentiment score between 0 and 10. (float)
    """

    if not isinstance(text, str) or not text.strip():
        return np.nan

    model.eval()

    # Tokenize full reflection without special tokens
    token_ids = tokenizer(
        text,
        add_special_tokens=False
    )["input_ids"]

    if len(token_ids) == 0:
        return np.nan

    # Reserve positions for [CLS] and [SEP]
    chunk_size = max_length - 2

    chunks = [
        token_ids[i:i + chunk_size]
        for i in range(
            0,
            len(token_ids),
            chunk_size
        )
    ]

    weighted_scores = []
    total_tokens = 0

    for chunk in chunks:

        # Add BERT special tokens manually
        input_ids = (
            [tokenizer.cls_token_id]
            + chunk
            + [tokenizer.sep_token_id]
        )

        attention_mask = (
            [1] * len(input_ids)
        )

        inputs = {
            "input_ids": torch.tensor(
                [input_ids],
                dtype=torch.long,
                device=device
            ),
            "attention_mask": torch.tensor(
                [attention_mask],
                dtype=torch.long,
                device=device
            )
        }

        with torch.no_grad():
            outputs = model(**inputs)

        probability_positive = F.softmax(
            outputs.logits,
            dim=-1
        )[0][1].item()

        chunk_length = len(chunk)

        weighted_scores.append(
            probability_positive
            * chunk_length
        )

        total_tokens += chunk_length

    if total_tokens == 0:
        return np.nan

    weighted_probability = (
        sum(weighted_scores)
        / total_tokens
    )

    # Convert probability [0, 1] to score [0, 10]
    return weighted_probability * 10


def predict_konlpy_sentiment(
    reflection,
    model,
    tokenizer,
    okt,
    stopwords,
    max_len
):
    """
    Generate a 0-10 sentiment score for a full reflection
    using the KoNLPy baseline.

    The reflection is divided into sentences and each sentence
    receives a sentiment probability. Sentence probabilities
    are combined using token-weighted averaging.

    Parameters
    ----------
    reflection : Full daily reflection. (str)

    model
        Trained BiLSTM sentiment model.

    tokenizer
        Trained Keras tokenizer.

    okt
        KoNLPy Okt tokenizer.

    stopwords : Korean stopwords excluded during preprocessing. (list)

    max_len : Maximum sequence length used by the BiLSTM model. (int, default=128)

    Returns
    -------
    Sentiment score between 0 and 10. (float)
    """

    if (
        not isinstance(reflection, str)
        or not reflection.strip()
    ):
        return np.nan

    sentences = [
        sentence.strip()
        for sentence in re.split(
            r"[.!?]",
            reflection
        )
        if sentence.strip()
    ]

    if len(sentences) == 0:
        return np.nan

    weighted_scores = []
    total_tokens = 0

    for sentence in sentences:

        tokens = [
            word
            for word in okt.morphs(sentence)
            if word not in stopwords
        ]

        if len(tokens) == 0:
            continue

        sequence = (
            tokenizer.texts_to_sequences(
                [tokens]
            )
        )

        padded_sequence = pad_sequences(
            sequence,
            maxlen=max_len
        )

        probability_positive = float(
            model.predict(
                padded_sequence,
                verbose=0
            )[0][0]
        )

        token_count = len(tokens)

        weighted_scores.append(
            probability_positive
            * token_count
        )

        total_tokens += token_count

    if total_tokens == 0:
        return np.nan

    weighted_probability = (
        sum(weighted_scores)
        / total_tokens
    )

    return round(
        weighted_probability * 10,
        2
    )