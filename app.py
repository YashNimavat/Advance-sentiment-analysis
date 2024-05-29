import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pinecone
import torch
import seaborn as sns
import os
import time

from sentence_transformers import SentenceTransformer
from pinecone import ServerlessSpec
from pinecone import Pinecone

st.write(
    """
    ## Welcome to the hotel manegment system
    """
)
st.caption('Follow the step-by-step instruction to compare customer’s sentiments about the different hotels :sunglasses:')


# set device to GPU if available
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def init_retrieve_model():

    # load the model from huggingface
    retriever = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2',
        device=device
    )
    return retriever






#-----------------old 2023 method---------------------------------#
    # pinecone.init(
    #     api_key=str(os.environ['PINECONE_API_KEY']), 
    #     environment=str(os.environ['PINECONE_ENV']) 
    # )
    # index = pinecone.Index(index_name=os.environ['PINECONE_INDEX_NAME'])

    # connect to sentiment-mining index we created
    # index = Pinecone.Index(index_name)
#-----------------old 2023 method---------------------------------#

   
retriever = init_retrieve_model()

#------------------------------------------------#
# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = str(os.environ['PINECONE_API_KEY'])

# configure client
pc = Pinecone(api_key=api_key)

cloud = os.environ.get('Pinecone_CLOUD') or 'aws'
region = os.environ.get('Pinecone_REGION') or 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)


index_name = 'sentiment-mining'

existing_indexes = [
        index_info["name"] for index_info in pc.list_indexes()
]
index = pc.Index(index_name)


# check if the sentiment-mining index exists
if index_name not in existing_indexes:
        # create the index if it does not exist
    Pinecone.create_index(
        index_name,
        dimension=384,
        metric="cosine"
    )

        # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)
time.sleep(1)
# view index stats
print("Index Name: ",index)
print("Index stats:",index.describe_index_stats())

#------------------------------------------------#
query = 'Give me hotel list'
xq = retriever.encode(query).tolist()
# query pinecone
result = index.query(xq, top_k=500, include_metadata=True)

if 'stage' not in st.session_state:
    st.session_state.stage = 0

def set_state(i):
    st.session_state.stage = i

if st.session_state.stage == 0:
    st.button('Begin', on_click=set_state, args=[1])

if st.session_state.stage >= 1:
    # set device to GPU if available
    hotel_set = set()
    for i in range(0,len(result["matches"])):
        hotel_name = result["matches"][i]["metadata"]["hotel_name"]
        hotel_set.add(hotel_name)

    hotel_selection = st.multiselect(
        'Please select hotels',list(hotel_set),on_change=set_state, args=[2])
    hotels = hotel_selection
    # st.button('Confirm hotels selection', on_click=set_state, args=[2])

if st.session_state.stage >= 2:
    options = st.multiselect(
    'Select parameters which you want to compare',
    ['Room Size', 'Cleanliness', 'Staff', 'AC','Value'],
    ['Room Size'])

    unique_value = 0
    queries = {}
    query_list=[]
    for i in range(len(options)):
        st.write('please enter query regarding', options[i])
        unique_value += 1

        if options[i] == 'Room Size':
            options_query=st.text_input("e.g., are customers happy with the room sizes?",key=unique_value)

        elif options[i] == 'Cleanliness':
            options_query=st.text_input("e.g., are customers satisfied with the cleanliness of the rooms?",key=unique_value)
        elif options[i] == 'Staff':
            options_query=st.text_input("e.g., did the customers like how they were treated by the staff?",key=unique_value)
        elif options[i] == 'AC':
            options_query=st.text_input("e.g., customer opinion on the AC",key=unique_value)
        elif options[i] == 'Value':
            options_query=st.text_input("e.g., whether the price paid for the hotel is worth it?",key=unique_value)
        
        query_list.append(options_query)
        queries = dict(zip(options,query_list))

    st.button('Generate graph', on_click=set_state, args=[3])

if st.session_state.stage >= 3:
    def count_sentiment(result):
    # store count of sentiment labels
        sentiments = {
            "negative": 0,
            "neutral": 0,
            "positive": 0,
        }
        # iterate through search results
        for r in result["matches"]:
            # extract the sentiment label and increase its count
            sentiments[r["metadata"]["label"]] += 1
        return sentiments


    hotel_sentiments = []

    # iterate through the hotels
    for hotel in hotels:
        result = []
        # iterate through the keys and values in the queries dict
        for area, query in queries.items():
            # generate query embeddings
            xq = retriever.encode(query).tolist()
            # query pinecone with query embeddings and the hotel filter
            xc = index.query(xq, top_k=500, include_metadata=True, filter={"hotel_name": hotel})
            # get an overall count of customer sentiment
            sentiment = count_sentiment(xc)
            # sort the sentiment to show area and each value side by side
            for k, v in sentiment.items():
                data = {
                    "area": area,
                    "label": k,
                    "value": v
                }
                # add the data to result list
                result.append(data)
        # convert the
        hotel_sentiments.append({"hotel": hotel, "df": pd.DataFrame(result)})

    # create the figure and axes to plot barchart for all hotels
    fig, axs = plt.subplots(nrows=1, ncols=len(hotels), figsize=(25, 4.5))
    plot=plt.subplots_adjust(hspace=0.25)

    counter = 0
    # iterate through each hotel in the list and plot a barchart
    for d, ax in zip(hotel_sentiments, axs.ravel()):
        # plot barchart for each hotel
        sns.barplot(x="label", y="value", hue="area", data=d["df"], ax=ax)
        # display the hotel names
        ax.set_title(d["hotel"])
        # remove x labels
        ax.set_xlabel("")
        # remove legend from all charts except for the first one
        counter += 1
        if counter != 1: ax.get_legend().remove()
        # display the full figure
    # plt.show()
    plt.savefig('x',dpi=1000)
    st.image('x.png')
    os.remove('x.png')

    st.write('You can play around with different parameters for more insights')
    st.write('Thank you!')
    st.button('Start Over', on_click=set_state, args=[0])
