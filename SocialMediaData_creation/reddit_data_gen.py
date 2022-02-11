import pandas as pd
import pickle
import requests
import time
import datetime
import string
import praw
from typing import List
from tqdm import tqdm
import itertools
import argparse
from flashtext import KeywordProcessor

from reddit_utils import anonymize_text, remove_keywords

class RedditDataGen():
    """Scrapes and prepares reddit data."""

    def __init__(self, subreddit: str, max_submissions: int, sub_sort: str, sub_sort_type: str,
                 sub_after: int, sub_before: int, anonymize: bool):
        self.subreddit = subreddit
        self.max_submissions = max_submissions
        self.sub_sort = sub_sort
        self.sub_sort_type = sub_sort_type
        self.sub_after = sub_after
        self.sub_before = sub_before
        self.anonymize = anonymize
        self.url = "https://api.pushshift.io/reddit/search/submission"
        self.reddit_api = praw.Reddit(client_id='0oga1o2RGFHi0A',
                         client_secret='q54nEd4jFeJPZ9jNeIjYPoWqKx43gA',
                         user_agent='webscrape')

    def crawl_page(self, sub_before: int, last_page: List[dict] = None) -> List[dict]:
        """Crawls a page of results from a given subreddit using the Pushshift API.
        source: https://github.com/apatry/word-of-mouth/blob/master/wordofmouth/etl/reddit.py

        Args:
            sub_before (int): Epoch value for submission creation date. API will return
                                submissions before this date.
            last_page (List[dict], optional): The last downloaded page. Defaults to None.

        Raises:
            Exception: API server is inaccessible.

        Returns:
            List[dict]: The data in a page of results represented as a list of 
                        dictionaries, each dictionary contains information about one 
                        submission.
        """
        params = {"subreddit": self.subreddit, "size": 500,
                "sort": self.sub_sort, "sort_type": self.sub_sort_type,
                "after": self.sub_after, "before": sub_before}
        if last_page is not None:
            if len(last_page) > 0:
                # resume from where we left at the last page
                params["before"] = last_page[-1]["created_utc"]
            else:
                # the last page was empty, we are past the last page
                return []
        results = requests.get(self.url, params)
        if not results.ok:
            # something wrong happened
            raise Exception(
                "Server returned status code {}".format(results.status_code))
        return results.json()["data"]

    def get_submissions(self, sub_before: int = None, 
                        max_submissions: int = float("inf")) -> List[dict]:
        """Crawls submissions from a subreddit using the Pushshift API.
        source: https://github.com/apatry/word-of-mouth/blob/master/wordofmouth/etl/reddit.py

        Args:
            sub_before (int, optional): Epoch value for submission creation date. 
                                        API will return submissions before this date 
                                        (if specified). Defaults to None (API will 
                                        return submissions created until the day of 
                                        extraction).
            max_submissions (int, optional): Maximum number of submissions to be 
                                            scraped. 
                                            Defaults to float("inf"), i.e., no limit.

        Returns:
            List[dict]: A list of dictionaries, each dictionary contains information 
                        about one submission.
        """
        if sub_before is None:
            sub_before = self.sub_before

        submissions = []
        last_page = None   
        while last_page != [] and len(submissions) < max_submissions:
            last_page = self.crawl_page(sub_before, last_page)
            submissions += last_page
            time.sleep(3)
        
        if max_submissions == float("inf"):
            return submissions
        else:
            return submissions[:max_submissions]
    
    def get_all_submissions(self) -> List[dict]:
        """Fetches all submissions before the specified date (self.sub_before) until it
        hits the specified limit on maximum number of submissions (self.max_submissions).

        Returns:
            List[dict]: A list of dictionaries, each dictionary contains information 
                        about one submission.
        """
        sub_before = None
        all_submissions = []
        latest_submissions = self.get_submissions(sub_before)
        all_submissions += latest_submissions

        if self.max_submissions is not None:
            while latest_submissions != [] and len(all_submissions) < self.max_submissions:
                sub_before = latest_submissions[-1]["created_utc"]
                latest_submissions = self.get_submissions(
                                            sub_before,
                                            self.max_submissions - len(all_submissions)
                                            )
                all_submissions += latest_submissions
            return all_submissions[:self.max_submissions]
        
        else:
            while latest_submissions != []:
                sub_before = latest_submissions[-1]["created_utc"]
                latest_submissions = self.get_submissions(sub_before)
                all_submissions += latest_submissions
            return all_submissions

    def get_comments_for_sub(self, sub_id: str, comments_limit: int=None) -> List[dict]:
        """Gets comments for the submission with <sub_id>, using breadth-first traversal
        with PRAW API.

        Source:
        https://github.com/pistocop/subreddit-comments-dl/blob/master/src/subreddit_downloader.py

        Args:
           sub_id (str): ID of a submission to get comments for.
           comments_limit (int, optional): Number of "MoreComments" objects to replace.
                                            Replacing a "MoreComments" object is
                                            equivalent to clicking on the "MoreComments"
                                            button on the reddit webpage.
                                            Defaults to None, i.e., replace all
                                            "MoreComments" objects until there are none
                                            left.

        Returns:
            List[dict]: List of dictionaries. Each dictionary contains info about one
                        comment.
        """
        submission = self.reddit_api.submission(id=sub_id)
        submission.comments.replace_more(limit=comments_limit)
        comments = submission.comments.list()
        comments_lst = []
        for comment in comments:
            comment_info = {
                "id": comment.id,
                "link_id": comment.link_id,
                "body": comment.body,
                "created_utc": int(comment.created_utc),
                "depth": comment.depth,
                "comment_type": comment.comment_type,
                "ups": comment.ups,
                "downs": comment.downs,
                "likes": comment.likes,
                "author": comment.author,
                "permalink": comment.permalink,
            }

            comments_lst.append(comment_info)

        return comments_lst

    def keyword_search(self, df: pd.DataFrame, col_name: str, 
                       keyword_list: List[str]) -> pd.DataFrame:
        """
        Runs a keyword search over the input terms. Matches will be returned and added to
        the new column 'keyword'.

        Args:
            df (pd.DataFrame): DataFrame containing text to search.
            col_name (str): Name of the column that contains text to be
                            searched.
            keyword_list (List[str]): List of keywords to search.
        Returns:
            pd.DataFrame: df that has new keyword column for search hits.

        """
        kp = KeywordProcessor()
        kp.add_keywords_from_list(keyword_list)

        df['keywords_'+col_name] = df[col_name].apply(lambda x: kp.extract_keywords(x, span_info = True))

        return df

    def clean_and_filter_df(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Anonymizes usernames and removes keywords in col_name in df. Filters out
        removed/deleted/null entries in col_name. Creates submission/comment creation 
        date in UTC.

        Args:
            df (pd.DataFrame): DataFrame to clean and filter.
            col_name (str): Name of the column that contains text to be
                            cleaned and filtered.

        Returns:
            pd.DataFrame: df that has been cleaned and filtered.
        """
        if self.anonymize:
            df = df.apply(anonymize_text(col_name), axis=1)
            df = df[df.apply(remove_keywords, col_name=col_name, axis=1)]

        df = df[~df[col_name].isin(["[removed]", "[deleted]"])]
        df = df[~df[col_name].isnull()]
        df["created_utc_date"] = df["created_utc"].apply(
            lambda x: str(datetime.datetime.utcfromtimestamp(x))[:10]
            )

        return df

    def clean_and_filter_submissions(self, submissions: List[str]) -> pd.DataFrame:
        """Creates a DataFrame out of submissions List. Concatenates title and 
        body (selftext) columns of submissions to create a "text" column.
        Applies preliminary cleaning, and filters to a subset of useful columns in the 
        DataFrame. 

        Args:
            submissions (List[str]): A list of dictionaries, each dictionary contains 
                                    information about one submission.

        Returns:
            pd.DataFrame: DataFrame created from submissions.
        """

        df = pd.DataFrame(submissions)
        # Concatenate title and selftext (body of the post)
        df["text"] = df["title"] + " " + df["selftext"]
        # Clean and filter df
        df = self.clean_and_filter_df(df, "text")
        df = df[['id', 'author', 'created_utc', 'created_utc_date', 
                'selftext', 'title', 'text', 'domain', 'full_link',
                'is_original_content', 'is_video', 'num_comments', 'num_crossposts',
                'permalink', 'pinned', 'retrieved_on', 'score', 'subreddit',
                'subreddit_id', 'subreddit_subscribers', 'subreddit_type', 'title',
                'total_awards_received', 'treatment_tags', 'upvote_ratio']]

        return df

    def clean_and_filter_comments(self, comments: List[dict]) -> pd.DataFrame:
        """Creates a DataFrame out of comments List, and applies preliminary cleaning.

        Args:
            comments (List[dict]): List of dictionaries. Each dictionary contains info 
                                    about one comment.

        Returns:
            pd.DataFrame: DataFrame created out of comments.
        """

        df = pd.DataFrame(comments)
        df["author"] = df["author"].name
        df = self.clean_and_filter_df(df, "body")

        return df

    def merge_subs_comments(self, df_subs: pd.DataFrame,
                              df_comments: pd.DataFrame) -> pd.DataFrame:
        """Maps comments to their corresponding submissions.

        Args:
            df_subs (pd.DataFrame): Cleaned submissions DataFrame.
            df_comments (pd.DataFrame): Cleaned comments DataFrame.

        Returns:
            pd.DataFrame: A master DataFrame with submissions and comments.
        """
        df_subs = df_subs.copy()
        df_comments = df_comments.copy()
        df_subs.columns = ['sub_' + str(col) for col in df_subs.columns]
        df_comments.columns = ['comm_' + str(col) for col in df_comments.columns]
        
        df_comments["comm_sub_id"] = df_comments["comm_link_id"].str[3:]

        df_master = df_subs.merge(df_comments, how="left",
                                  left_on="sub_id", right_on="comm_sub_id")

        # Verify that there is no duplicate rows
        assert df_master.drop_duplicates(subset=["sub_id", "comm_id"]).shape[0] ==\
            df_master.shape[0]

        return df_master
    
    def concat_subs_comments(self, df_subs: pd.DataFrame, 
                                df_comments: pd.DataFrame) -> pd.DataFrame:
        """Filters df_subs and df_comments to a subset of useful columns ("id", 
        "author", "created_utc", "created_utc_date", "text"), adds a "type" column 
        indicating if an entry is a submiission or comment, concatenates these two 
        DataFrames and return it.

        Args:
            df_subs (pd.DataFrame): [description]
            df_comments (pd.DataFrame): [description]

        Returns:
            pd.DataFrame: [description]
        """
        df_comments.rename(columns={"body": "text"}, inplace=True)
        df_subs["type"] = "sub"
        df_comments["type"] = "comm"
        
        df = df_subs[
            ["id", "author", "created_utc", "created_utc_date", "text", "type"]].copy().append(
            df_comments[["id", "author", "created_utc", "created_utc_date", "text", "type"]].copy(), 
            ignore_index=True)
        return df


if __name__ == "__main__":
    # Add argparse
    parser = argparse.ArgumentParser(
        description="Pull submissions and comments from Reddit, output is one DataFrame \
            with submissions and comments")
    parser.add_argument("--subreddit", type=str, default="covidlonghaulers",
                        help="Name of the subreddit to pull from")
    parser.add_argument("--max_submissions", type=int,
                        help="Number of submissions to pull.")
    parser.add_argument("--pulls_comments", type=bool, default=True,
                        help="If True, program pulls specified number of submissions \
                            and all comments linked to them")
    parser.add_argument("--merged_filepath", type=str, default="merged_df.csv",
                        help=".csv file path for saving the DataFrame of comments \
                            joined with submissions")
    parser.add_argument("--concat_filepath", type=str, default="concat_df.csv",
                        help=".csv file path for saving the DataFrame of submissions \
                            concatenated with comments")
    parser.add_argument("--subs_filepath", type=str, default="subs_df.csv",
                        help=".csv file path for saving the DataFrame of only \
                            submissions")
    parser.add_argument("--comments_filepath", type=str, default="comments_df.csv",
                        help=".csv file path for saving the DataFrame of only comments")
    # Optional submission parameters
    # Source: https://reddit-api.readthedocs.io/en/latest/
    parser.add_argument("--submission_sort_type", type=str,
                        choices=["score", "num_comments", "created_utc"],
                        default="created_utc",
                        help="Sort submissions search by a specific attribute")
    parser.add_argument("--submission_sort", type=str,
                        choices=["asc", "desc"],
                        default="desc",
                        help="Sort submissions search results in a specific order")
    parser.add_argument("--submission_after", type=int,
                        default=None,
                        help="Return submissions after this date, enter \
                        epoch value for created_utc.")
    parser.add_argument("--submission_before", type=int,
                        default=None,
                        help="Return submissions before this date, enter \
                        epoch value for created_utc.")
    parser.add_argument("--search_terms", type=str,
                        default="",
                        help="""String of keywords separated by commas (no spaces) to \
                        search within the input data. E.g. "long haul,longhauler,long covid" """)
    parser.add_argument("--anonymize", action="store_true", default=True,
                        help="Whether to anonymize the reddit data")
    args = parser.parse_args()

    subreddit = args.subreddit
    max_submissions = args.max_submissions
    pulls_comments = args.pulls_comments
    merged_filepath = args.merged_filepath
    concat_filepath = args.concat_filepath
    subs_filepath = args.subs_filepath
    comments_filepath = args.comments_filepath
    # Optional submission parameters
    sub_sort_type = args.submission_sort_type
    sub_sort = args.submission_sort
    sub_after = args.submission_after
    sub_before = args.submission_before
    search_terms = args.search_terms
    anonymize = args.anonymize
    
    # Pull data from reddit
    reddit_data = RedditDataGen(subreddit, max_submissions, sub_sort, sub_sort_type,
                 sub_after, sub_before, anonymize)
    # Pull submissions
    all_submissions = reddit_data.get_all_submissions()
    pickle.dump(all_submissions, open("reddit_sub_interm.pkl", "wb"))
    df_subs = reddit_data.clean_and_filter_submissions(all_submissions)
    assert df_subs["id"].nunique() == df_subs.shape[0]

    # look for search terms
    if search_terms:
        df_subs = reddit_data.keyword_search(df_subs, 'text', search_terms.split(","))

    if pulls_comments:
        # Pull comments
        latest_comments = []
        for i in tqdm(df_subs['id'].unique(),
                    desc="Getting comments for each submission"):
            latest_comments.append(reddit_data.get_comments_for_sub(i))
        latest_comments_lst = list(itertools.chain.from_iterable(latest_comments))
        df_comm = reddit_data.clean_and_filter_comments(latest_comments_lst)
        assert df_comm["id"].nunique() == df_comm.shape[0]

        if search_terms:
            df_comm = reddit_data.keyword_search(df_comm, 'body', search_terms.split(","))

        # Merge submissions and comments and save output
        df_merged = reddit_data.merge_subs_comments(df_subs, df_comm)
        df_merged.to_csv(merged_filepath, index=False)
        
        # Concat submissions and comments and save output
        df_concat = reddit_data.concat_subs_comments(df_subs, df_comm)
        df_concat.to_csv(concat_filepath, index=False)

        # Save comments and submissions separately
        df_subs.to_csv(subs_filepath, index=False)
        df_comm.to_csv(comments_filepath, index=False)

        # Simple checks
        print("Shape of merged DataFrame (with submissions and comments): ",
              df_merged.shape)
        print("Shape of submissions DataFrame (number of unique submissions): ",
              df_subs.shape)
        print("Shape of comments DataFrame (number of unique comments): ", df_comm.shape)
        print("Number of unique submissions with comments: ",
            df_merged[df_merged["comm_id"].notnull()]["sub_id"].nunique())

    else:
        # Save submissions as output
        df_subs.to_csv(subs_filepath, index=False)

        # Simple checks
        print("Shape of submissions DataFrame (number of unique submissions): ",
              df_subs.shape)

