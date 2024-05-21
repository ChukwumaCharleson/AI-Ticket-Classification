#Pinecone team has been making a lot of changes to there code and here is how it should be used going forward :)
from pinecone import Pinecone as PineconeClient
#from langchain.vectorstores import Pinecone     #This import has been replaced by the below one :)
from langchain_community.vectorstores import Pinecone
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
#from langchain.llms import OpenAI #This import has been replaced by the below one :)
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
#from langchain.callbacks import get_openai_callback #This import has been replaced by the below one :)
from langchain_community.callbacks import get_openai_callback
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_community.llms import Ollama
import joblib
from pages.admin_utils2 import *
from langchain.schema.messages import HumanMessage as hm, SystemMessage, AIMessage



def create_embedding():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

def define_rag_chain():
    llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2")
    #llm = Ollama(model="llama2")
    #llm = ChatAnthropic(model="claude-2")
    vectorstore = Chroma(persist_directory='db', embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))
    retriever = get_retriever(vectorstore)
    #vectorstore = push_to_chroma(docs_chunks,embeddings)

    ### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone single question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is. \
    
    
    For example:

    If user asks: does the HR document contain?
    and you respond: It dontains details about the HR policy of the company

    and then the user asks again: What does it say in it's conclusion?

    Your resulting standalone question would be: What does the HR policy say in the conclusion of the HR policy?

    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    ### Answer question ###
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Do not attempt to complete the question to provide additional information \
    Your answers MUST be straight to the point and address only what was asked. 
  

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    
    ### Statefully manage chat history ###
    store = {}


    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    print(conversational_rag_chain)

    return conversational_rag_chain

def get_answer(user_input, rag_chain, chat_history):    
    ### Manage chat history ###
    
    ai_msg = rag_chain.invoke({"input": user_input},
    config={
        "configurable": {"session_id": "abc123"}
    },  )["answer"]
    #chat_history.extend([HumanMessage(content=user_input), ai_msg["answer"]])
    return ai_msg


def convert_message(m):
    if m["role"] == "user":
        return hm(content=m["content"])
    elif m["role"] == "assistant":
        return AIMessage(content=m["content"])
    elif m["role"] == "system":
        return SystemMessage(content=m["content"])
    else:
        raise ValueError(f"Unknown role {m['role']}")



def _format_chat_history(chat_history):
    def format_single_chat_message(m):
        if type(m) is hm:
            return "Human: " + m.content
        elif type(m) is AIMessage:
            return "Assistant: " + m.content
        elif type(m) is SystemMessage:
            return "System: " + m.content
        else:
            raise ValueError(f"Unknown role {m['role']}")

    return "\n".join([format_single_chat_message(m) for m in chat_history])


def predict(query_result):
    Fitmodel = joblib.load('modelsvm.pk1')
    result=Fitmodel.predict([query_result])
    return result[0]