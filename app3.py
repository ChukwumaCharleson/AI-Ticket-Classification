from dotenv import load_dotenv
import streamlit as st
import os
from user_utils2 import *


#Creating session variables
if 'HR_tickets' not in st.session_state:
    st.session_state['HR_tickets'] =[]
if 'IT_tickets' not in st.session_state:
    st.session_state['IT_tickets'] =[]
if 'Transport_tickets' not in st.session_state:
    st.session_state['Transport_tickets'] =[]


def main():
    rag_chain = define_rag_chain()
    load_dotenv()
    input_chat_history = []

    st.title("AI Assistant and Ticket Classification Tool")
    st.write("By Chukwuma Charleson Onwuka")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # render older messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    #Capture user input
    
    user_input = st.chat_input("Enter your message...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        # render the user's new message
        with st.chat_message("user"):
            st.markdown(user_input)

        # render the assistant's response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            with st.spinner(text="Thinking..."):
                  
                    assistant_response = get_answer(user_input, rag_chain)
                    print(assistant_response)
                    message_placeholder.write(assistant_response)
                    #message_placeholder.write(assistant_response)

        #Button to create a ticket with respective department
        button = st.button("Submit ticket?")

        if button:
            #Get Response
            print("Computing embeddings")
            embeddings = create_embedding()

            print("Applying embeddings to query result")
            query_result = embeddings.embed_query(user_input)
            print("Applied embeddings successfully")
            print(query_result)

            #loading the ML model, so that we can use it to predict the class to which this complaint belongs
            print("Predicting value")
            
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



