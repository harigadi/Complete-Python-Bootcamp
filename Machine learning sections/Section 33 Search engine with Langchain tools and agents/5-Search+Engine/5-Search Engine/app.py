import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

## Arxiv and wikipedia Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

search=DuckDuckGoSearchRun(name="Search")


st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API Key:",type="password")


# st.session_state["messages"] stores the conversation history.
# If no history exists, it seeds one assistant message: "Hi, I'm a chatbot who can search the web..."

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assisstant","content":"Hi,I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display existing messages
# Iterates through st.session_state.messages and renders each as a chat message.

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# User input

# When the user submits text via st.chat_input(...), it adds the prompt to the
#  session state and displays the user message.

# Agent creation and response

# Creates a new ChatGroq LLM each time with streaming enabled.
# Builds search_agent using ZERO_SHOT_REACT_DESCRIPTION, so it can decide which tool to use.
# Runs the agent on the full message history.
# StreamlitCallbackHandler shows the agent’s thoughts live in the UI.
# The final response is appended to the conversation and displayed.

# What the walrus operator := does
# It is called the assignment expression.
# It lets you assign a value to a variable as part of a larger expression.
# In your code
# This means:

# Call st.chat_input(...)
# Assign its result to prompt
# Test the value of prompt in the if condition
# So it is equivalent to:
# prompt = st.chat_input(placeholder="What is machine learning?")
# if prompt:

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True)
    tools=[search,arxiv,wiki]

    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)

## What this block does

# - `with st.chat_message("assistant"):`  
#   - Opens a Streamlit chat message bubble for the assistant.
#   - Anything rendered inside this block appears as the assistant's reply.

# - `st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)`  
#   - Creates a callback handler that streams the agent’s internal reasoning and tool actions into the app.
#   - `st.container()` gives it a place to render the progress inside the assistant message.
#   - `expand_new_thoughts=False` keeps new reasoning steps collapsed by default.

# - `response = search_agent.run(st.session_state.messages, callbacks=[st_cb])`  
#   - Runs the LangChain agent on the current conversation history.
#   - The agent can use the search/Wikipedia/ArXiv tools as needed.
#   - The callback handler streams the agent’s thought process into the UI while the agent runs.
#   - `response` is the final text answer produced by the agent.

# - `st.session_state.messages.append({'role':'assistant',"content":response})`  
#   - Saves the returned assistant text into the conversation history.
#   - This lets future turns include the assistant’s reply.

# - `st.write(response)`  
#   - Displays the final assistant response text inside the same chat bubble.

# ### In short

# This code:
# 1. creates an assistant chat bubble,
# 2. runs the agent with streaming callbacks,
# 3. saves the agent’s reply to history,
# 4. and shows the reply in the UI.

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)

