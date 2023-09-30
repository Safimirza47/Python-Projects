import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
import pandas as pd
import vertexai
from vertexai.language_models import TextGenerationModel, InputOutputTextPair, ChatModel
from vertexai.preview.language_models import TextGenerationModel

#Read dataset
df = pd.read_excel("chatbot data1.xlsx")
dictexcel = pd.read_excel("datase_refined.xlsx")

#create example dictionary, with user answers as keys and sentiments as values
sent_dict = dict(zip(df.iloc[::3]['input_text'], df.iloc[::3]['Sentiment']))
chat_dict = dict(zip(dictexcel['input_text'], df['output_text']))

# App title
st.set_page_config(page_title="🌴💬 LLM Powered Smartphone Survey Chatbot")

# Replicate Credentials
with st.sidebar:
    st.title('🌴💬 Welcome to SurveySage!')
    st.markdown('''
    ## About
    This is an AI assisted, LLM-powered survey chatbot built using:
    - [Google Cloud Platform](https://cloud.google.com)
    - [Palm 2 language model](https://ai.google/discover/palm2/)
    - [Streamlit](https://streamlit.io)
    
    💡 Note: Prototype chat version.
    ''')
    add_vertical_space(4)
    st.write('Made with ❤️ by Team VMA (University of Leicester)')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hey there! I'm SurveySage, an AI powered smartphone survey tool. I will be your buddy for the next few minutes.  Let's start with your name and how do you feel today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hey there! I'm SurveySage, an AI powered smartphone survey tool. I will be your buddy for the next few minutes.  Let's start with your name and how do you feel today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

#write function to call LLM for sentiment analysis
def predict_sentimentLLM(
    model_name: str,
    temperature: float,
    max_output_tokens: int,
    top_p: float,
    top_k: int,
    content: str,
    ) :
    model1 = TextGenerationModel.from_pretrained("text-bison@001")
    model1 = model1.get_tuned_model(model_name)
    response1 = model1.predict(
        content,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_k=top_k,
        top_p=top_p,)
    return response1.text

# Response Generation
def responsegenerator(
    model_name2: str,
    temperature: float,
    max_output_tokens: int,
    top_p: float,
    top_k: int,
    new_content: str,
    ) :
    model2 = TextGenerationModel.from_pretrained("text-bison@001")
    model2 = model2.get_tuned_model(model_name2)
    response2 = model2.predict(
        new_content,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_k=top_k,
        top_p=top_p,)
    return response2.text

context = ""

# User-provided prompt
prompt = ""
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            multi_line_string = "\n".join([f"input: {key}\nsentiment: {value}\n" for key, value in sent_dict.items()])
            additional_string = "\ninput: {prompt}\nsentiment: "
            multi_line_string+=additional_string
            content = multi_line_string.format(prompt=prompt)
            response_text = predict_sentimentLLM(
                "projects/117401557332/locations/us-central1/models/6759980006497058816",
                temperature=0.2, 
                max_output_tokens=5, 
                top_p=0.8, 
                top_k=1, 
                content=content)
            if response_text == "positive":
                context = "You are a survey chatbot named VMA. You will ask questions based on the user's answers. Generate a question with positive sentiment and add happy emotions to the generated question. Focus on the subject of Smartphones and avoid asking about anything else. This is important to maintain the relevance and quality of your question. Do not repeat the generated question."
            elif response_text == "negative":
                context = "You are a survey chatbot named VMA. You will ask questions based on the user's answers. Generate a question to a negative-sentiment filled input from user, such that the generated question cheers up the user. Focus on the subject of Smartphones and avoid asking about anything else. This is important to maintain the relevance and quality of your question. Do not repeat the generated question."
            elif response_text == "neutral":
                context = "You are a survey chatbot named VMA. You will ask questions based on the user's answers. Generate a questions to a neutral-sentiment filled user input, and add happy emotions to the generated questions. Focus on the subject of Smartphones and avoid asking about anything else. This is important to maintain the relevance and quality of your questions. Do not repeat the generated question."
                    
            new_content = f"""{context}
            
            input: Hi my name is Midhun. I feel fantastic!
            output: Hi Midhun! That is very exciting to hear. Glad that you are having a fun day. What's your age?
        
            input: I am 26 years old
            output: Alright!  Do you want to mention your gender if that's okay?
        
            input: I am male
            output: Thanks! What smartphone brand do you use. Can you also specify the model of your phone?
        
            input: I use an Apple iPhone
            output: Wow that\'s fancy. How long have you been using this phone of yours?
        
            input: I've been using it for almost 2 years now.
            output: That is reasonable. What are some of the common tasks that you perform on your smartphone?
                
            input: {prompt}
            output: 
            """
            generated_text = responsegenerator(
                model_name2 = "projects/117401557332/locations/us-central1/models/1204754681763463168",
                temperature=0.2,
                max_output_tokens=256,
                top_p=0.8,
                top_k=40,
                new_content = new_content)
            
            placeholder = st.empty()
            full_response = ''
            for item in generated_text:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
                    
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
