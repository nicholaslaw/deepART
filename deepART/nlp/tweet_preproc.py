import os
import json
import pandas as pd
import logging
import html
import re
import concurrent.futures
from .text_preproc import *

logger = logging.Logger('__name__')

default_fields = ['id','text','lang','timestamp_ms','entities','user','in_reply_to_status_id','in_reply_to_user_id']
default_preprocs = {"html_unescape": None, "remove_patterns": None, "replace_apo":None, "replace_slang":None, "remove_shorts":"default"}

class Tweet_preprocesser:

    def __init__(self, fields =default_fields, preprocs = default_preprocs):
        self.fields = fields
        self.preprocs = preprocs

    def set_path(self, path, batch_size = 0):
        '''
        set the folder path that containes the tweets
        '''
        self.path = path
        self.batch_size = batch_size
        self.files = []

        try:
            names = os.listdir(self.path)
        except Exception as e:
            logger.debug(e)

        for name in names:
            if os.path.isfile(self.path+'/'+name):
                self.files.append(self.path+'/'+name)
        

    def _load_data(self, tweets_data_path, batch_size):
        '''
        generator to load mini-batches of data
        '''
        count = 0
        tweets_data = []
        with open(tweets_data_path, "r") as tweets_file:
            while(True):
                tweets_data = []
                count = 0
                for line in tweets_file:
                    try:
                        tweet = json.loads(line)
                        tweets_data.append(tweet)
                        count += 1
                        #break up into batches and yield the data for processing
                        if count >= batch_size and batch_size !=0:
                            break
                    except Exception as e:
                        pass

                if len(tweets_data) == 0:
                    break
                else:
                    yield tweets_data
        
        return


    def _get_fields(self, tweets_data):
        '''
        extract fields specified by self.fields from a batch twitter jsons
        '''
        tweets_list = []
        row = 0
        for tweet_json in tweets_data:
            tweet_dict = {}
            if tweet_json['lang'] == 'en':
                for field in self.fields:
                    tweet_dict[field] = tweet_json[field]
                tweets_list.append(tweet_dict)

        return pd.DataFrame(tweets_list)


    def process(self,prefix="tweets", output_path = None, nthreads=0):
        '''
        process tweets based on config specified
        '''
        if not output_path==None:
            try:
                os.mkdir(output_path)
            except FileExistsError:
                logger.info("Directory exists...")

        #create threading workers
        if nthreads> 0:
            main_executor = concurrent.futures.ThreadPoolExecutor(max_workers=nthreads)

        batch_num = 0
        for file in self.files:
            data_gen = self._load_data(file, self.batch_size)

            
            for tweets_data in data_gen:
                self.tweets = self._get_fields(tweets_data)
                if nthreads<1:
                    #self._unescape_html(input_field='text')
                    self._batch_remPatterns(input_field='text')
                    self._batch_repApo(input_field='clean_tweet')
                    self._batch_repSlang(input_field='clean_tweet')

                else:
                    self._batch_remPatterns(input_field='text', threading=main_executor)
                    self._batch_repApo(input_field='clean_tweet',threading=main_executor)
                    self._batch_repSlang(input_field='clean_tweet',threading=main_executor)


            #output clean tweets
            batch_num+=1
            if output_path == None:
                yield self.tweets.copy()
            else:
                try:
                    self.tweets.to_csv(path_or_buf="{}/{}-{:08d}.csv".format(output_path,prefix,batch_num))
                    logger.info("Current batch:".format(batch_num))
                except Exception as e:
                    logger.debug(e)

        return 1
    
    def _unescape_html(self,input_field = 'text', output_field = 'clean_tweet'):
        '''
        unescape html for the a batch of tweets
        '''
        self.tweets[output_field]=self.tweets[input_field].apply(lambda tweet: html.unescape(tweet))
        
        return
    
    
    def _batch_remPatterns(self,patterns = None, input_field = 'text', output_field = 'clean_tweet',threading=None):
        '''
        remove specified patterns for a batch of tweets
        '''
        if threading is not None:
            future_clean = {threading.submit(lambda tweet: remove_patterns(tweet), tweet): (i,tweet) for i, tweet in enumerate(self.tweets[input_field])}
            for future in concurrent.futures.as_completed(future_clean):
                index = future_clean[future][0]
                try:
                    clean_tweet = future.result()
                except Exception as exc:
                    logger.debug('%r generated an exception: %s' % (index, exc))
                else:
                    self.tweets.at[index, output_field] = clean_tweet
        else:
            self.tweets[output_field]=self.tweets[input_field].apply(lambda tweet: remove_patterns(tweet))

        return
    


    def _batch_repApo(self,apo_dict = None, input_field = 'text', output_field = 'clean_tweet', threading=None):
        '''
        remove specified patterns for a batch of tweets
        '''
        if threading is not None:
            future_clean = {threading.submit(lambda tweet: replace_apo(tweet), tweet): (i,tweet) for i, tweet in enumerate(self.tweets[input_field])}
            for future in concurrent.futures.as_completed(future_clean):
                index = future_clean[future][0]
                try:
                    clean_tweet = future.result()
                except Exception as exc:
                    logger.debug('%r generated an exception: %s' % (index, exc))
                else:
                    self.tweets.at[index, output_field] = clean_tweet
        else:
            self.tweets[output_field]=self.tweets[input_field].apply(lambda tweet: replace_apo(tweet))

        return
        

    def _batch_repSlang(self,slang_dict = None, input_field = 'text', output_field = 'clean_tweet',threading=None):
        '''
        replace slangs for a batch of tweets
        '''
        if threading is not None:
            future_clean = {threading.submit(lambda tweet: replace_slang(tweet), tweet): (i,tweet) for i, tweet in enumerate(self.tweets[input_field])}
            for future in concurrent.futures.as_completed(future_clean):
                index = future_clean[future][0]
                try:
                    clean_tweet = future.result()
                except Exception as exc:
                    logger.debug('%r generated an exception: %s' % (index, exc))
                else:
                    self.tweets.at[index, output_field] = clean_tweet
        else:
            self.tweets[output_field]=self.tweets[input_field].apply(lambda tweet: replace_slang(tweet))

        return
    

