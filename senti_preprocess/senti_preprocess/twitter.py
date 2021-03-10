import re
from itertools import zip_longest
import pandas


def remove_urls(tweet_string):
    # return re.sub(r"^https?:\/\/.*[\r\n]*", " ", tweet_string)
    return re.sub(r"http\S+", " ", tweet_string)


def remove_mentions(tweet_string):
    return re.sub(r"@[A-Za-z0-9]+", " ", tweet_string)


def segment_hashtags(tweet_string):
    hashtags = re.findall(r"#(\w+)", tweet_string)

    for hashtag in hashtags:
        segmented_hashtag = ""
        for letter, letter_next in zip_longest(hashtag, hashtag[1:], fillvalue=""):
            if letter.isupper() and not any([letter_next.isupper(), letter_next == ""]):
                segmented_hashtag += " " + letter
            else:
                segmented_hashtag += letter
        tweet_string = re.sub(hashtag, segmented_hashtag, tweet_string)

    return tweet_string.replace("#", "")
