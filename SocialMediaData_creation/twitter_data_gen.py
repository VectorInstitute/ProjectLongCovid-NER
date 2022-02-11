import hashlib
import json
import pathlib
from copy import deepcopy
from past.types import unicode
from searchtweets import gen_request_parameters, load_credentials, ResultStream

from tqdm import tqdm

from demographic_inferer import DemoInferer

class TwitterDataGen:
    """
    Provides the ability to scrape tweets via the Twitter API v2.
    """
    def __init__(self,
                 credentials_file,
                 credentials_key,
                 run_config,
                 output_file,
                 processing_passes=None,
                 infer_demographic=False):
        self.credentials = load_credentials(filename=credentials_file,
                                            yaml_key=credentials_key,
                                            env_overwrite=False)
        self.output_file = pathlib.Path(output_file)

        # Get config
        self.run_config = run_config
        self.hashtags = self.run_config.get("keywords", "hashtags").split(", ")
        self.word_include = self.run_config.get("keywords",
                                                "include").split(", ")
        self.words_exclude = self.run_config.get("keywords",
                                                 "exclude").split(", ")

        self.pre_req = self.run_config.get("keywords",
                                            "pre_req").split(", ")

        self.language = self.run_config.get("fields", "language").split(", ")

        # Get requested tweet fields
        tweet_fields = run_config.get("fields", "tweet_fields")
        self.tweet_fields_string = tweet_fields.replace(" ", "")

        # Get requested user fields
        user_fields = run_config.get("fields", "user_fields")
        self.user_fields_string = user_fields.replace(" ", "")

        self.start_time = run_config.get("dates", "start")
        self.end_time = run_config.get("dates", "end")
        self.n_results = run_config.get("num-results", "num")

        self.infer_demographic = infer_demographic
        if self.n_results != "all":
            self.n_results = int(self.n_results)
            if self.n_results < 10:
                raise ValueError(
                    "Library does not support requests less than 10 tweets")
        else:
            # some arbitrarily large number
            self.n_results = int(10e10)

        # Generate tweet query string
        self.query_string = self._generate_tweet_query_string(
            print_query_string=True)

        # Generate tweet query
        self.query = self._generate_tweet_request(print_query=True)

        # Generate response generator
        self.response = self._send_tweet_request()

        # demographic inferer
        self.demo_inferer = DemoInferer()
        # register pass for tweet cleaning
        self.registered_passes = []
        if processing_passes is not None:
            for f in processing_passes:
                if not callable(f):
                    raise TypeError(
                        "function {} in processing_passes must be a callable".
                        format(f))
            self.registered_passes.append(f)

    def __iter__(self):
        is_final_batch = False

        # Batches tweets in groups of 500 until there is <500 tweets left
        batched_tweets = []
        for entry in tqdm(self.response, total=int(self.n_results)):

            # next token info, we don't need as the api handles pagination for us
            if "newest_id" in entry:
                continue

            # every few hundred entries we get a single dump of information, merge
            # and yield
            if "users" in entry:
                users = entry["users"]
                merged_batched = self._merge_user_and_tweet(
                    batched_tweets, users)
                cleaned_merged_batched = self._clean_tweets(merged_batched)
                batched_tweets = []
                yield cleaned_merged_batched

            elif "id" in entry:
                batched_tweets.append(entry)

            else:
                raise ValueError(
                    "Unknown entry encountered during response\n{}".format(
                        entry))

    def register_pass(self, processing_pass):
        if not callable(processing_pass):
            raise TypeError(
                "function {} in processing_pass must be a callable".format(
                    processing_pass))

        self.registered_passes.append(processing_pass)

    def _generate_tweet_query_string(self, print_query_string=False):
        query_string = ""

        if self.pre_req[0] != '':
            pre_req = "({})".format(" OR ".join(self.pre_req))
        else:
            pre_req = ""
        # Adding hashtags to query_string
        if self.hashtags[0] != '':
            self.hashtags = ["#" + s for s in self.hashtags]
        query_string += " OR ".join(self.hashtags)

        if self.hashtags[0] != '' and self.word_include[0] != "":
            query_string += " OR "

        # Adding word_include's, add "OR" if not empty list
        if self.word_include[0] != '':
            self.word_include = ['"{}"'.format(s) for s in self.word_include]
        query_string += " OR ".join(self.word_include)



        if len(query_string) == 0:
            raise ValueError("Must specify a hashtag or an exact match")

        # Adding brackets around hashtags and includes, as expected behavior for Twitter API
        query_string = f"{pre_req} ({query_string})"

        # Adding word_exclude's, add " -" if not empty list
        if self.words_exclude[0] != "":
            query_string += " -"
        query_string += " -".join(self.words_exclude)

        # Adding language preference
        query_string += f" lang:{self.language[0]}"

        # Checking query_string length
        if len(query_string) > 1024:
            raise ValueError(
                f"Query string length ({len(query_string)}) cannot be greater than 1024"
            )

        if print_query_string:
            print(f"Query String: {query_string}")

        return query_string

    def _generate_tweet_request(self, print_query=False):
        """
         Default tweet fields are "id" and "text" and will be include automatically. For the full list of options,
         see https://developer.twitter.com/en/docs/twitter-api/data-dictionary/object-model/tweet.

         Defaults user fields are "id", "name", and "username" and will be included automatically. For the full list
         of options, see https://developer.twitter.com/en/docs/twitter-api/data-dictionary/object-model/user.
        """

        query = gen_request_parameters(
            self.query_string,
            start_time=self.start_time,
            end_time=self.end_time,
            tweet_fields=self.tweet_fields_string,
            # user_fields=user_fields_string, doesn't work for some reason
            results_per_call=100)

        query = json.loads(query)
        # Requests user fields manually
        query["expansions"] = "author_id"
        query["user.fields"] = f"{self.user_fields_string}"
        addtional_info = ['profile_image_url']
        if "description" not in self.user_fields_string:
            addtional_info.append("description")
        query["user.fields"] += ","+ \
                                ",".join(addtional_info)

        query = json.dumps(query)
        if print_query:
            print(f"Query: {query}")

        return query

    def _send_tweet_request(self, print_response=False):

        response = ResultStream(request_parameters=self.query,
                                max_tweets=self.n_results,
                                **self.credentials)

        response = response.stream()
        if print_response:
            print("Response:")
            for tweet in response:
                print(tweet)

        return response

    def _merge_user_and_tweet(self, tweets, users):
        """
        Tweets are separated from user data when received from the Twitter API. This method merges the
        data together.
        """
        if self.infer_demographic:
            # Convert users into map to quicken user assignment
            inferer_inputs = self.demo_inferer.get_inferer_inputs(users)
            inferer_outputs = self.demo_inferer.infer(inferer_inputs)

            assert len(inferer_inputs) == len(inferer_outputs)

        users_dict = {user["id"]: user for user in users}

        # Assigns user data to each tweet in tweet_dict
        for tweet in tweets:
            uid = tweet["author_id"]
            if self.infer_demographic:
                users_dict[uid]["demographic_info"] = inferer_outputs[uid]
            tweet["user"] = users_dict[uid]

        return tweets

    def _clean_tweets(self, tweets, print_response=False):
        cleaned_tweets = []
        for i, tweet in enumerate(tweets):
            tweet = deepcopy(tweet)
            keep = True
            for func in self.registered_passes:
                tweet = func(tweet)
                if tweet is None:
                    keep = False
                    break
            if keep:
                cleaned_tweets.append(tweet)

        if print_response:
            print("Response:")
            for tweet in tweets:
                print(tweet)

        return cleaned_tweets

    def json_dump(self, contents, print_response=False):

        if self.output_file.exists():
            with open(self.output_file, "r+") as file:
                data = json.load(file)
                data.append(contents)
                file.seek(0)
                json.dump(data, file, indent=4, sort_keys=True)

        else:
            with open(self.output_file, "w") as file:
                json.dump(contents, file, indent=4, sort_keys=True)

        if print_response:
            print(f"Written to {self.output_file}:")
            print(json.dumps(contents, indent=4, sort_keys=True))
