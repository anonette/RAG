import streamlit as st
import os
from langchain_utils import initialize_chat_conversation
from search_indexing import download_and_index_pdf
import re
#keys loaded using streamlit secrets
# ~/.streamlit/secrets.toml


# Page title
st.set_page_config(page_title='RAG2023',menu_items={'Get Help':"https://github.com/anonette/RAG.git",'About': "check denisa kera's work at https://anonette.net"} )
st.title('RAG for research')
st.subheader('Discourse analysis assistant')

# Initialize the faiss_index key in the session state. This can be used to avoid having to download and embed the same PDF
# every time the user asks a question
if 'faiss_index' not in st.session_state:
    st.session_state['faiss_index'] = {
        'indexed_urls': [],
        'index': None
    }

# Initialize conversation memory used by Langchain
if 'conversation_memory' not in st.session_state:
    st.session_state['conversation_memory'] = None

# Initialize chat history used by StreamLit (for display purposes)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Store the URLs added by the user in the UI
if 'urls' not in st.session_state:
    st.session_state.urls = []

if 'active_urls' not in st.session_state:
    st.session_state.active_urls = []

def remove_url(url_to_remove):
    """
    Remove URLs from the session_state. Triggered by the respective button
    """
    if url_to_remove in st.session_state.urls:
        st.session_state.urls.remove(url_to_remove)

def save_urls_to_file():
    with open('urls.txt', 'w') as file:
        for url in st.session_state.urls:
            file.write(url + '\n')

if 'urls' not in st.session_state:
    st.session_state.urls = []
    if os.path.exists('urls.txt'):
        with open('urls.txt', 'r') as file:
            st.session_state.urls = [line.strip() for line in file.readlines()]
           
with st.sidebar:

    #https://commission.europa.eu/system/files/2020-02/commission-white-paper-artificial-intelligence-feb2020_en.pdf
    #https://commission.europa.eu/document/download/d2ec4039-c5be-423a-81ef-b9e44e79825b_en?filename=commission-white-paper-artificial-intelligence-feb2020_en.pdf
    
    # Add/Remove URLs form
    with st.form('urls-form', clear_on_submit=True):
        url = st.text_input('URLs to relevant PDFs: ', value='https://eur-lex.europa.eu/resource.html?uri=cellar:e0649735-a372-11eb-9585-01aa75ed71a1.0001.02/DOC_1&format=PDF')
        add_url_button = st.form_submit_button('Add')
        if add_url_button:
            if url not in st.session_state.urls:
                st.session_state.urls.append(url)
                save_urls_to_file() 

    # Display a container with the URLs added by the user so far
    with st.container():
        if st.session_state.urls:
            st.header('URLs added:')
            for url in st.session_state.urls:
                st.write(url)
                st.button(label='Remove', key=f"Remove {url}", on_click=remove_url, kwargs={'url_to_remove': url})
                st.divider()
    with st.container():
        st.markdown('''
                    project by denisa kera at [anonette.net](https://anonette.net)  
                    code at [anonette/RAG.git](https://github.com/anonette/RAG.git)
                    ''')
                      
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if query_text := st.chat_input("Your message"):

    # Display user message in chat message container, and append to session state
    st.chat_message("user").markdown(query_text)
    st.session_state.messages.append({"role": "user", "content": query_text})

    # Check if FAISS index already exists, or if it needs to be created as it includes new URLs
    session_urls = st.session_state.urls
    if st.session_state['faiss_index']['index'] is None or set(st.session_state['faiss_index']['indexed_urls']) != set(session_urls):
        st.session_state['faiss_index']['indexed_urls'] = session_urls
        with st.spinner('Downloading and indexing PDFs...'):
            faiss_index = download_and_index_pdf(session_urls)
            st.session_state['faiss_index']['index'] = faiss_index
    else:
        faiss_index = st.session_state['faiss_index']['index']

    # Check if conversation memory has already been initialized and is part of the session state
    if st.session_state['conversation_memory'] is None:
        conversation = initialize_chat_conversation(faiss_index)
        st.session_state['conversation_memory'] = conversation
    else:
        conversation = st.session_state['conversation_memory']

    # Search PDF snippets using the last few user messages
    search_number_messages = 4     # Number of past user messages that will be used to search relevant snippets
    user_messages_history = [message['content'] for message in st.session_state.messages[-search_number_messages:] if message['role'] == 'user']
    user_messages_history = '\n'.join(user_messages_history)

    with st.spinner('Querying LLM...'):
        response = conversation.predict(input=query_text, user_messages_history=user_messages_history)
        # Append the response to a file 
        with open('response.txt', 'a') as file:
            file.write(response + '\n')
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
        snippet_memory = conversation.memory.memories[1]
        for page_number, snippet in zip(snippet_memory.pages, snippet_memory.snippets):
            with st.expander(f'Snippet from page {page_number + 1}'):
                # Remove the <START> and <END> tags from the snippets before displaying them
                snippet = re.sub("<START_SNIPPET_PAGE_\d+>", '', snippet)
                snippet = re.sub("<END_SNIPPET_PAGE_\d+>", '', snippet)
                st.markdown("> "+snippet)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
