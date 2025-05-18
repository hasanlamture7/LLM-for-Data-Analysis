# streamlit_app.py
import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
import os

# Load .env if available
load_dotenv()

# Set OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

st.title("📊 LLM-Powered Data Visualization App")

# Check for API key
if not openai_api_key:
    st.error("🚨 Missing OpenAI API key. Please set the `OPENAI_API_KEY` environment variable.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("### 🧾 Preview of Data")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    # User input
    question = st.text_input("Ask a question about your data (e.g., 'Plot total sales over time'):")

    if question:
        st.write("🤖 Analyzing your data...")

        try:
            # Create LangChain agent
            agent = create_pandas_dataframe_agent(
                ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=openai_api_key),
                df,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                allow_dangerous_code=True,
            )

            with st.spinner("Running analysis..."):
                response = agent.run(question)
                st.write("### 📈 Result")
                st.write(response)
        except Exception as e:
            st.error(f"❌ Error during code execution: {e}")
