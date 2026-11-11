import numpy as np 
import re
from typing import Any, List, Dict, Union
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime


def jaccard_similarity(a, b):
    """Compute the Jaccard similarity between two sets or lists.
    
    Args:
    a, b : set or list
        Input sets or lists to compare.
    
    Returns:
    float
        Jaccard similarity (between 0 and 1)
    """
    set_a = set(a)
    set_b = set(b)
    
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    
    if not union:
        return 0.0  
    
    return len(intersection) / len(union)


def normalize(arr:List, smooth:float = 0):
    """Normalize a list or array so that its elements sum to 1.

    Args:
        arr (List): Input list or array of numerical values.
        smooth (float, optional): A smoothing constant added to each element 
            before normalization. Defaults to 0.

    Returns:
        np.ndarray: Normalized array where the elements sum to 1. 
        If the sum of the array is 0, returns the original array unchanged.
    """
    if np.sum(arr) > 0:
        return (np.array(arr)+smooth)/np.sum(np.array(arr)+smooth)
    else:
        arr


def extract_subreddit_from_permalink(input_str):
    """Extract the subreddit name from a Reddit permalink string.

    Args:
        input_str (str): A permalink or path string that may contain a 
            subreddit reference (e.g., "/r/AskReddit/comments/...").

    Returns:
        str: The subreddit name in lowercase without the "/r/" prefix. 
        Returns an empty string if no subreddit is found in the input.
    """
    match = re.match(r"(/r/[^/]+)", input_str)
    x = ""
    if match:
        x = str(match.group(1))[3:].lower()
    return x


def utc2datetime(timestamp):
    """Convert a UTC timestamp into a datetime object.

    The function interprets the input as seconds since the Unix epoch
    (January 1, 1970) and returns a timezone-naive datetime object
    representing the corresponding UTC time.

    Args:
        timestamp (float): UTC timestamp in seconds since the Unix epoch.

    Returns:
        datetime: Datetime object representing the given UTC timestamp.
    """
    dt = datetime.utcfromtimestamp(timestamp)
    return dt


def datetime2utc(dt):
    """Convert a datetime object into a UTC timestamp.

    The function converts the input datetime into the number of seconds
    since the Unix epoch (January 1, 1970). The result is rounded to
    the nearest second.

    Args:
        dt (datetime): Datetime object to convert. Can be naive (assumed UTC)
            or timezone-aware.

    Returns:
        int: UTC timestamp in seconds since the Unix epoch.
    """
    timestamp = round(int(float(dt.timestamp())))
    return timestamp