### Imports
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import pinecone
import boto3
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import langchain
import os
import json
import os
import sys
from datetime import datetime
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()
from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path="langchain.db")


### --- Housekeeping ---
pinecone_api_key = st.secrets["pinecone_api_key"]
pinecone_environment = st.secrets["pinecone_environment"]
ACCESS_KEY = st.secrets["aws_access_key_id"]
SECRET_KEY = st.secrets["aws_secret_access_key"]

client = boto3.client('bedrock',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name='us-east-1')
bedrockrt = boto3.client('bedrock-runtime',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name='us-east-1')

module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock

boto3_bedrock = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None),
)
model_E = 'amazon.titan-embed-text-v1'
model_L = "anthropic.claude-v2"
cl_llm = Bedrock(
    model_id=model_L,
    client=boto3_bedrock,
    model_kwargs={'temperature': 0.2, "max_tokens_to_sample": 50000},
)

conversation = ConversationChain(
    llm=cl_llm, verbose=False
)
accept = 'application/json'
contentType = 'application/json'

pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
index = pinecone.Index('dfc')

### - Search Function --
def collect_messages(input):
    global claude_prompt, booktext, prompt
    prompt = input
    if prompt == "":
        return
    
    body = json.dumps({
                "inputText": prompt
            })

    response = bedrockrt.invoke_model(body=body, modelId=model_E, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')

    myoutput = index.query(
        vector=embedding,
        filter={
            "document": {"$eq": mydemo+".pdf"}  # "Blueprint.pdf" "NIST.pdf" "OMB_Memo.pdf"
        },
        top_k=40,
        include_metadata=True
    )

    booktext = ""
    for x in myoutput['matches']:
        if x['score']>.2:
            booktext += x['metadata']['text']+"\n"
            
    claude_prompt = PromptTemplate.from_template("""

Human: You are an AI Bot that answers questions as if you were the author of the documents.
You will use the information in between the HTML tags <document_text> to answer the user's question.
Provide a very detailed answer that is at least 500 characters long. 
If the AI does not know the answer to a question or if the section between the HTML tags <document_text> 
is blank, then respond with the phrase "I don't know."
DO NOT embed your response in any HTML tags, such as <document_text> or </document_text>. 

Document Text:
<document_text>
"""+booktext+"""
</document_text>

Here is the human's next reply:
<human_reply>
{input}
</human_reply>

Assistant:
""")

    conversation.prompt = claude_prompt
        
    response = conversation.predict(input=prompt)
 
    return response


### - Layout components --
## I put these at the top because Streamlit runs from the top down and 
## I need a few variables that get defined here. 

## Layout configurations
st.set_page_config(
    page_title='AI Document App', 
    layout="wide",
    initial_sidebar_state='collapsed',
)
## CSS is pushed through a markdown configuration.
## Streamlit layout is not flexible. It's good for internal apps, not so good for customer facing apps.
padding_top = 10
st.markdown(f"""
    <style>
        .block-container, .main {{
            padding-top: {padding_top}px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

## UI Elements starting with the Top Graphics
col1, col2 = st.columns( [1,5] )
col1.image('AderasBlue2.png', width=50)
col1.image('AderasText.png', width=50)
col2.title('Interactive Document Application')
st.markdown('---')
## Add a sidebar
with st.sidebar: 
    mydemo = st.selectbox('Select Document', ['OMB_Memo', 'Blueprint', 'NIST'])
    st.markdown("---")
    tz = st.container()


### Main code block
    
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Ask away...?"):
    start = datetime.now()
    tz.write("Start: "+str(start))
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Processing..."):
        response = collect_messages(input=prompt)

    st.session_state.messages.append({"role": "assistant", "content": response})    
    st.chat_message('assistant').write(response)

    tz.write("End: "+str(datetime.now()))
    tz.write("Duration: "+str(datetime.now() - start))
