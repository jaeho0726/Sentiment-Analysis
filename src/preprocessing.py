import numpy as np
import pandas as pd


def clean_nsmc_data(data):
    """
    Clean the NSMC dataset.

    Parameters
    ----------
    data : NSMC dataframe containing 'document' and 'label' columns. (pandas.DataFrame)

    Returns
    -------
    Cleaned NSMC dataframe. (pandas.DataFrame)
    """

    data = data.copy()

    # Remove duplicate reviews
    data.drop_duplicates(
        subset=["document"],
        inplace=True
    )

    # Remove missing values
    data = data.dropna(how="any")

    # Keep Korean characters and spaces
    data["document"] = data["document"].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)

    # Replace empty reviews with NaN
    data["document"] = data["document"].replace("", np.nan)

    # Remove empty reviews
    data = data.dropna(how="any")

    return data.reset_index(drop=True)


def prepare_journal_data(data):
    """
    Clean and format the daily reflection dataset.

    Parameters
    ----------
    data : Raw journal dataframe imported from Google Sheets. (pandas.DataFrame)

    Returns
    -------
    Cleaned journal dataframe. (pandas.DataFrame)
    """

    data = data.copy()

    # Shorten column names
    data = data.rename(
        columns={
            "Work Intensity (0 - Easy / 10 - Intense)": "Work Intensity",
            "Overall Mood (0 - Poor / 10 - Great)": "Overall Mood",
            "Name": "Date"
        }
    )

    # Convert numeric columns
    data["Hours of Work"] = pd.to_numeric(
        data["Hours of Work"],
        errors="coerce"
    ).fillna(0)

    data["Work Intensity"] = pd.to_numeric(
        data["Work Intensity"],
        errors="coerce"
    ).fillna(0)

    # Missing mood should remain missing rather than becoming 0 b/c 0 implies negative mood in mood score
    data["Overall Mood"] = pd.to_numeric(
        data["Overall Mood"],
        errors="coerce"
    )

    # Convert and sort dates
    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    data = data.sort_values(
        by="Date"
    ).reset_index(drop=True)

    data["Day_of_Week"] = (
        data["Date"].dt.day_name()
    )

    return data