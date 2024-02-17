#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Hamza Farooq
"""

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
import os
nlp = spacy.load("en_core_web_sm")
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import datetime

from spacy import displacy
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib import pyplot as plt

import nltk
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
from tqdm.autonotebook import tqdm
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import scipy.spatial
import pickle
from sentence_transformers import SentenceTransformer, util
import torch





# import utils as utl

import time
import torch
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration
from string import punctuation
# tr = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import scipy.spatial


from sentence_transformers import SentenceTransformer, util
import torch



def main():




    # Settings
    st.set_page_config(layout="wide", page_title='Paris Hotel Finder', page_icon="🎈"   )
    from string import punctuation
    punctuation=punctuation+ '\n'


    from sentence_transformers import SentenceTransformer, util
    import torch
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    import scipy.spatial

    from sentence_transformers import SentenceTransformer, util
    import torch
    #import os
    @st.cache(allow_output_mutation=True)
    def load_model():
        return SentenceTransformer('all-MiniLM-L6-v2'),SentenceTransformer('multi-qa-MiniLM-L6-cos-v1'),CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    embedder,bi_encoder,cross_encoder = load_model()




    #original_title = '<p style="font-family:IBM Mono; color:Blue; font-size: 20px;">Original image</p>'
    st.title("travelle - Parisian Hotel Finder")
    with st.expander("ℹ️ - About this app", expanded=True):

        st.write(
            """
    -   travelle is a hotel search engine that allows users to enter free text query to make the search result personalized to user preference as opposed to other travel websites where a user has to spend hours going through hotel list.
    -   We use natural language processing and big data to return results customized for your preferences.
    -   A user can enter just about anything and we will narrow the results to what closely matches your requirements.
    -   For e.g. a user can enter a query like "Hotel near the Eiffel and cheaper than $300 per night with free breakfast" and we will find the closest results
    	    """
        )


    punctuation=punctuation+ '\n'


    #import os

    # embedder = SentenceTransformer('all-MiniLM-L6-v2')



    def lower_case(input_str):
        input_str = input_str.lower()
        return input_str

    df_all = pd.read_csv('paris_clean_newer.csv')


    df_combined = df_all.sort_values(['Hotel']).groupby('Hotel', sort=False).text.apply(''.join).reset_index(name='all_review')
    df_combined_paris_summary = pd.read_csv('df_combined_paris.csv')
    df_combined_paris_summary = df_combined_paris_summary[['Hotel','summary']]

    import re

    # df_combined = pd.read_csv('df_combined.csv')

    df_combined['all_review'] = df_combined['all_review'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))


    df_combined['all_review']= df_combined['all_review'].apply(lambda x: lower_case(x))
    df_basic = df_all[['Hotel','description','price_per_night']].drop_duplicates()
    df_basic = df_basic.merge(df_combined_paris_summary,how='left')
    df_combined_e = df_combined.merge(df_basic)
    df_combined_e['all_review'] =df_combined_e['description']+ df_combined_e['all_review'] + df_combined_e['price_per_night']

    df = df_combined_e.copy()


    df_sentences = df_combined_e.set_index("all_review")

    df_sentences = df_sentences["Hotel"].to_dict()
    df_sentences_list = list(df_sentences.keys())



    import pandas as pd
    from tqdm import tqdm
    from sentence_transformers import SentenceTransformer, util

    df_sentences_list = [str(d) for d in tqdm(df_sentences_list)]
    #
    corpus = df_sentences_list
    # corpus_embeddings = embedder.encode(corpus,show_progress_bar=True)
    corpus_embeddings = np.load('embeddings.npy')

    bi_encoder.max_seq_length = 512     #Truncate long passages to 256 tokens
    top_k = 32                          #Number of passages we want to retrieve with the bi-encoder

    #The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality

    # corpus_embeddings_h = np.load('embeddings_h_r.npy')

    with open('corpus_embeddings_bi_encoder.pickle', 'rb') as pkl:
        doc_embedding = pickle.load(pkl)

    with open('tokenized_corpus.pickle', 'rb') as pkl:
        tokenized_corpus = pickle.load(pkl)

    bm25 = BM25Okapi(tokenized_corpus)
    passages = corpus




# We lower case our text and remove stop-words from indexing
    def bm25_tokenizer(text):
        tokenized_doc = []
        for token in text.lower().split():
            token = token.strip(string.punctuation)

            if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
                tokenized_doc.append(token)
        return tokenized_doc


    def search(query):
        # q = [str(userinput)]
        doc = nlp(str(userinput))

        ent_html = displacy.render(doc, style="ent", jupyter=False)
# Display the entity visualization in the browser:
        st.markdown(ent_html, unsafe_allow_html=True)
        ##### BM25 search (lexical search) #####
        bm25_scores = bm25.get_scores(bm25_tokenizer(query))
        top_n = np.argpartition(bm25_scores, -5)[-5:]
        bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
        bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)

        bm25list = {}
        st.title("Top-5 lexical search (BM25) hits")
        for hit in bm25_hits[0:5]:
            row_dict = df.loc[df['all_review']== corpus[hit['corpus_id']]]

            st.subheader(row_dict['Hotel'].values[0])
            de = df_basic.loc[df_basic.Hotel == row_dict['Hotel'].values[0]]
            st.write(f'\tPrice Per night: {de.price_per_night.values[0]}')
            st.write(f'Description: {de.description.values[0]}')
            st.expander(de.description.values[0],expanded=False)
            # try:
            #     st.write('Summary')
            #     st.expander(de.summary.values[0],expanded=False)
            # except:
            #     None
            # doc = corpus[hit['corpus_id']]
            # kp.get_key_phrases(doc)

            bm25list[row_dict['Hotel'].values[0]] = de.description.values[0][0:200]

        #### Sematic Search #####
        # Encode the query using the bi-encoder and find potentially relevant passages
        question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    #     question_embedding = question_embedding.cuda()
        hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
        hits = hits[0]  # Get the hits for the first query

        ##### Re-Ranking #####
        # Now, score all retrieved passages with the cross_encoder
        cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
        cross_scores = cross_encoder.predict(cross_inp)

        # Sort results by the cross-encoder scores
        for idx in range(len(cross_scores)):
            hits[idx]['cross-score'] = cross_scores[idx]

        # Output of top-5 hits from bi-encoder
        st.write("\n-------------------------\n")
        st.title("Top-5 Bi-Encoder Retrieval hits")
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)
        for hit in hits[0:5]:
    #         st.write("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']].replace("\n", " ")))
            row_dict = df.loc[df['all_review']== corpus[hit['corpus_id']]]
            st.subheader(row_dict['Hotel'].values[0])
            de = df_basic.loc[df_basic.Hotel == row_dict['Hotel'].values[0]]
            st.write(f'\tPrice Per night: {de.price_per_night.values[0]}')
            st.write(f'Description: {de.description.values[0]}')
            st.expander(de.description.values[0])
            # try:
            #     st.write('Summary')
            #     st.expander(de.summary.values[0],expanded=False)
            # except:
            #     None

        # Output of top-5 hits from re-ranker
        st.write("\n-------------------------\n")
        st.title("Top-5 Cross-Encoder Re-ranker hits")
        hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
        for hit in hits[0:5]:
    #         st.write("\t{:.3f}\t{}".format(hit['cross-score'], passages[hit['corpus_id']].replace("\n", " ")))
            row_dict = df.loc[df['all_review']== corpus[hit['corpus_id']]]
            st.subheader(row_dict['Hotel'].values[0])
            de = df_basic.loc[df_basic.Hotel == row_dict['Hotel'].values[0]]
            st.write(f'\tPrice Per night: {de.price_per_night.values[0]}')
            st.write(f'Description: {de.description.values[0]}')
            st.expander(de.description.values[0])
            # try:
            #     st.write('Summary')
            #     st.expander(de.summary.values[0],expanded=False)
            # except:
            #     None




    sampletext = 'e.g. Hotel near Eiffel Tower with big rooms'
    userinput = st.text_input('Tell us what are you looking in your hotel?','e.g. Hotel near Eiffel Tower with big rooms',autocomplete="on")
    da = st.date_input(
        "Date Check-in",
        datetime.date(2023, 6, 3))

    dst = st.date_input(
        "Date Check-out",
        datetime.date(2023, 6, 8))


    if not userinput or userinput == sampletext:
        st.write("Please enter a query to get results")
    else:
        query = [str(userinput)]
        doc = nlp(str(userinput))
        search(str(userinput))

        # We use cosine-similarity and torch.topk to find the highest 5 scores

if __name__ == '__main__':
    main()
