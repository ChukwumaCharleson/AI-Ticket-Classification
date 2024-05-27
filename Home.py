__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from dotenv import load_dotenv
import streamlit as st
import os
from user_utils import *


#Creating session variables
if 'HR_tickets' not in st.session_state:
    st.session_state['HR_tickets'] =[]
if 'IT_tickets' not in st.session_state:
    st.session_state['IT_tickets'] =[]
if 'Transport_tickets' not in st.session_state:
    st.session_state['Transport_tickets'] =[]
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'current_input' not in st.session_state:
    st.session_state.current_input = ''

def display_how_to_use():
    st.sidebar.title("How to Use")
    st.sidebar.markdown("### [GitHub Repository](https://github.com/ChukwumaCharleson/Automatic-Ticket-Classification-Tool)")
    st.sidebar.markdown("""
    1. **Load Data Store**: Upload a PDF document containing relevant information. Sample data can be found on my github repo above.

    2. **Create ML Model**: Upload training data to train, validate, and save a machine learning model capable of predicting the appropriate department for each question (A sample of the training data is available on my github repo if retraining is required).

    3. **Home**: Engage with the chat interface to ask questions about the uploaded PDF. If the bot doesn't have the answer, click "Submit Ticket" to utilize the trained ML model for department prediction.

    4. **Pending Tickets**: View submitted tickets and track their status in the Pending Tickets section.
    """)

def main():
    load_dotenv()

    st.title("AI Assistant and Ticket Classification Tool")
    st.write("By Chukwuma Charleson Onwuka")
    st.write("Please check the 'How to Use' section in the sidebar for instructions and sample documents.")

    display_how_to_use()
    
    # render older messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    #Capture user input
    user_input = st.chat_input("Enter your message...")
    rag_chain = define_rag_chain()
    input_chat_history = []
    user_input_backup = user_input

    if user_input:
        st.session_state.current_input = user_input
        st.session_state.messages.append({"role": "user", "content": user_input})

        # render the user's new message
        with st.chat_message("user"):
            st.markdown(user_input)

        # render the assistant's response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            with st.spinner(text="Thinking..."):
                  
                    assistant_response = get_answer(user_input, rag_chain, input_chat_history)
                    print(assistant_response)
                    message_placeholder.write(assistant_response)

        # Add assistant response to chat history            
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_response,
            }
        )

    #Button to create a ticket with respective department
    button = st.button("Submit ticket?")

    if button:
        user_input = st.session_state.current_input
        #Get Response
        print("Computing embeddings")
        embeddings = create_embedding()

        print("Applying embeddings to query result")

        query_result = embeddings.embed_query(user_input)
        
        department_value = predict(query_result)
        st.write("your ticket has been sumbitted to : "+department_value)

        #Appending the tickets to below list, so that we can view/use them later on...
        if department_value=="HR":
            st.session_state['HR_tickets'].append(user_input)
        elif department_value=="IT":
            st.session_state['IT_tickets'].append(user_input)
        else:
            st.session_state['Transport_tickets'].append(user_input)

if __name__ == '__main__':
    main()



