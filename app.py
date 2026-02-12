import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchResults
from langchain_classic.agents import initialize_agent,AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()

st.title("Groq Search Engine")

wiki_wrap=WikipediaAPIWrapper(doc_content_chars_max=250,top_k_results=1)
arxiv_wrap=ArxivAPIWrapper(doc_content_chars_max=250,top_k_results=1)
wiki=WikipediaQueryRun(api_wrapper=wiki_wrap)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrap)


search=DuckDuckGoSearchResults(name="Search")

st.sidebar.write("Settings")
api_key=st.sidebar.text_input("Enter your api key",type="password")

if "messages" not in st.session_state:
    st.session_state['messages']=[
        {"role":"assistant","content":"Hi! I am a chatbot who can search the web"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)


    llm=ChatGroq(model="llama-3.1-8b-instant",groq_api_key=api_key,streaming=True)

    tools=[wiki,arxiv,search]

    agent=initialize_agent(tools=tools,llm=llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True,
                           max_iterations=5,
    max_execution_time=30,
    agent_kwargs={
        "prefix": "Use tools only when strictly necessary. Answer directly if possible."
    })

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=agent.run(prompt,callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)





    
