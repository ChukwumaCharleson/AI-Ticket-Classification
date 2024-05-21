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


def main():
    load_dotenv()
    #os.environ["HUGGINGFACEHUB_API_TOKEN"]
    #os.environ["ANTHROPIC_API_KEY"]

    st.header("AI Assistant and Ticket Classification Tool")
    #Capture user input
    st.write("We are here to help you, please ask your question:")
    user_input = st.text_input("🔍")
    rag_chain = define_rag_chain()
    input_chat_history = []

    if user_input:

        #This will return the fine tuned response by LLM
        response= get_answer(user_input, rag_chain, input_chat_history)
        #input_chat_history = new_history
        st.write(response)

        #Button to create a ticket with respective department
        button = st.button("Submit ticket?")

        if button:
            #Get Response
            
            embeddings = create_embedding()
            query_result = embeddings.embed_query(user_input)

            #loading the ML model, so that we can use it to predit the class to which this compliant belongs to...
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



