from typing import List, Any, Dict
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import requests
import time

# HEADERS = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
# }

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
TEXTS_TO_CLEAN = [
    "reddit - the heart of the internet ",
    "skip to main content open menu open navigation go to reddit home ",
    "log in log in to reddit expand user menu open settings menu browse other communities ",
]

CATEGORIES_MAP = {
    
    "is banned you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "is banned for violating reddit's content policy you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "this subreddit has been banned. you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "this subreddit was banned due to a violation of our content policy (https://www.reddit.com/help/contentpolicy/). you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "this subreddit was banned due to a violation of our content policy (https://www.reddit.com/help/contentpolicy/) you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "this subreddit was banned due to a violation of our [content policy] (https://www.reddit.com/help/contentpolicy/). you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "this subreddit was banned due to a violation of our [content policy] (https://www.reddit.com/help/contentpolicy/) you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "this subreddit has been banned for violating the reddit rules to keep everyone safe you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "this subreddit was banned due to a violation of our content policy. you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "this subreddit has been banned you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "banned for violating reddit's content policy . you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "this subreddit was banned due to being in violation of our content policy. you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "this subreddit was banned due to violations of the reddit content policy . you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "this subreddit was banned due to violating our content policy. you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "this subreddit was banned due to a violation of our user agreement . you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "this subreddit has been banned for violations of reddit's site-wide rules. you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "this community has been banned for violations of reddit's site-wide content policies. you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "this subreddit has been banned for violating the reddit rules to keep everyone safe . you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "banned for multiple violations of site wide rules. you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "this subreddit was banned for breaking the rules of reddit . you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "banned for violating our content policy: https://www.reddit.com/help/contentpolicy/ you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "is banned . you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "this community has been banned for violating the reddit user agreement you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "this community has been banned. you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "this subreddit has been banned for violating the rules of reddit . you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "is banned this subreddit was banned. you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "this subreddit was banned due to a violation of our content policy you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "this subreddit was banned for breaking the rules of reddit you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "this community has been banned you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "this subreddit has been banned for violations of reddit's content policy . you agree that use of this site constitutes acceptance of":"no_specific_reason",
    "this subreddit was banned due to terms of use violations in regards to the reddit content policy . if you'd like to take it over please make a post":"no_specific_reason",
    "internal server error":"no_specific_reason",
    "is banned new moderators may request control of this subreddit via":"no_specific_reason",

    "creating or repurposing a sub to reconstitute":"regulation_evasion",
    "creating or repurposing a subreddit to reconstitute":"regulation_evasion",
    "ban evasion":"regulation_evasion",
    "quarantine evasion":"regulation_evasion",
    "attempting to evade restrictions imposed on an existing reddit community":"regulation_evasion",    
    "enforcement evasion":"regulation_evasion",    
    "the remaking of previously banned communities":"regulation_evasion",  
    "as a recreation of a previously banned subreddit":"regulation_evasion",  
    "being used for be":"regulation_evasion",  # BE: ban evasion

    "infringement policy":"copyright",
    "excessive copyright removals":"copyright",
    "excessive copyright takedowns":"copyright",
    "copyright repeat infringer policy":"copyright",
    "copyright infringement":"copyright",
    "copyright violation":"copyright",
    "trademark infringement":"copyright",

    "posting personal information":"impersonating",
    "impersonating someone in a misleading or deceptive manner":"impersonating",
    "breaking reddit or doing anything that interferes with normal use of reddit":"interfere_normal_use",
    "interfering with normal use of the site by utilizing css click hijacking":"interfere_normal_use",
    
    "you may not use reddit to break the law":"illegal_activity",
    "illegal":"illegal_activity",
    "violating rule 7":"illegal_activity",
    
    "for more information, please see: http://www.reddit.com/r/blog/comments/pmj7f/a_necessary_change_in_policy/":"sexual_content",
    "against minor sexualization":"sexual_content",
    "against sexual or suggestive content involving minors or someone who appears to be a minor":"sexual_content",
    "sexual or suggestive content involving minors":"sexual_content",
    "being used for som":"sexual_content",  # SOM: sexualization of minors
    "minor sexualisation":"sexual_content",
    "the sexualization of minors":"sexual_content",
    # "involuntary pornography":"sexual_content",
    "involuntary porn":"sexual_content",
    "non-consensual intimate media":"sexual_content",
    "rule 4":"sexual_content",
    "directed at underage users":"sexual_content",
    
    "the proliferation of personal and confidential information":"privacy",
    "inciting the posting of personal information":"privacy",

    "transactions involving prohibited goods or services":"prohibited_transaction",
    "facilitating the transaction of illegal goods":"prohibited_transaction",
    "the sales of prohibited goods":"prohibited_transaction",
    "being used for marketplace":"prohibited_transaction",
    "violations of reddit's marketplace policy":"prohibited_transaction",
    "transactions of prohibited goods":"prohibited_transaction",
    "the marketplace policy":"prohibited_transaction",
    "fraudulent services":"prohibited_transaction",

    "banned for spam":"spam",
    "being used for spam":"spam",
    "rule 2": "spam", # can be others but just to select the most common type in this category: https://redditinc.com/policies/reddit-rules 

    "being unmoderated":"unmoderated",
    "lack of moderators":"unmoderated",
    "lack of moderation":"unmoderated",
    "unmoderation":"unmoderated",
    "moderator code of conduct":"unmoderated",

    "harassing content":"harassment_or_violence",
    "posting of content that harasses or bullies":"harassment_or_violence",
    "harassment":"harassment_or_violence",
    "inciting harm against others":"harassment_or_violence",
    "our policy against bullying":"harassment_or_violence",
    "violent content":"harassment_or_violence",
    "violating rule 1, 2 and 8":"harassment_or_violence",
    "violence policy":"harassment_or_violence",
    "content that encourages or incites violence":"harassment_or_violence",
    "being used for violence":"harassment_or_violence",
    "for glorifying violence":"harassment_or_violence",
    "content that glorifies or encourages violence":"harassment_or_violence",
    "inciting violence":"harassment_or_violence",
    "encourages violence":"harassment_or_violence",
    "glorifying violence":"harassment_or_violence",
    "glorifying or encouraging violence":"harassment_or_violence",
    "encourages physical harm":"harassment_or_violence",
    "glorification of sexual violence":"harassment_or_violence",
    "hateful content":"harassment_or_violence",
    "encouraging violence and the dissemination of personal information":"harassment_or_violence",
    "the incitement to violence":"harassment_or_violence",
    "the glorification of violence":"harassment_or_violence",
    "incites violence":"harassment_or_violence",
    "incite violence":"harassment_or_violence",
    "against animal abuse":"harassment_or_violence",
    "the violence policy":"harassment_or_violence",
    "promoting hate":"harassment_or_violence",
    "violation rule 1":"harassment_or_violence",
    "violating rule 1":"harassment_or_violence",

    "vote manipulation":"vote_manipulation",
    "threatening the structural integrity of the greater reddit community":"threaten_integrity",
    
}

MANUL_CORRECT_MAP = {
    "amazondailydealsplus":"restricted",
}

def extract_reason(text):
    reason = text
    for phrase,cat in CATEGORIES_MAP.items():
        if phrase in text:
            reason = cat
            break
    return reason


def check_subreddit_status(subreddits:List[str], delay:int=2, print_raw:bool=False) -> Dict:
    """
    Check status for a list of subreddits (lite version).
    Returned HTML may not be a full version 
    --> compromised ability to detect restricted subreddits.

    Args: 
        subreddits (list of str): a list of subreddit names
        delay (int): seconds to wait between requests

    Returns:
        dict: subreddit -> html content text (to be further cleaned)
    """
    output = {}
    session = requests.Session()  # persistent session for cookies & connection reuse

    for i,sub in enumerate(subreddits):
        print("Checking subreddit:", i, sub)
        url = f"https://www.reddit.com/r/{sub}"

        try: 
            response = session.get(url, headers=HEADERS, timeout=5)

            # handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", delay))
                print(f"\tRate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                response = requests.get(url, headers=HEADERS, timeout=5)

            if print_raw:
                print(response.text)
            soup = BeautifulSoup(response.text, "html.parser")
            msg = soup.get_text(" ", strip=True)

            # detect restricted subreddit and add a tag in the msg
            details_div = soup.find("div", class_="community-details")
            if details_div:
                all_text = details_div.get_text(" ", strip=True)
                if "Restricted" in all_text:
                    msg += "[[ this subreddit is restricted ]]"

        except requests.exceptions.Timeout:
            msg = "timeout"
        except requests.exceptions.RequestException as e:
            msg = f"error: {e}"

        print("\tResult:", msg)
        output[sub] = msg
        time.sleep(delay)
    return output


def clean(text, subrname):
    for t in TEXTS_TO_CLEAN:
        text = text.replace(t, "")
    text = text.replace(f"{subrname.lower()} a chip a close button ", "")
    text = text.replace(subrname.lower(), "")
    return text


def check_status_from_message(text):
    status = "quiet"
    if "r/r/ is banned" in text:
        status = "banned"
    elif text == "internal server error":
        status = "banned"
    elif "this community is private" in text:
        status = "private"
    elif "doesn't have any posts yet" in text:
        status = "empty"
    elif "community not found" in text:
        status = "notfound"
    elif "is quarantined" in text:
        status = "quarantined"
    elif "[[ this subreddit is restricted ]]" in text:
        status = "restricted"
    return status