import streamlit as st
import pandas as pd
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from io import StringIO

# Page setup
st.set_page_config(
    page_title="UFO Sightings Database Enquiry",
    page_icon="🛸",
    layout="centered"
)

# Initialize OpenAI client
if 'client' not in st.session_state:
    st.session_state.client = None

# Sidebar for settings
with st.sidebar:
    st.subheader("Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        st.session_state.client = OpenAI(api_key=api_key)
        st.success("✓ API key set")
    else:
        st.warning("Please enter API key to continue")

# Data loading with full dataset processing
@st.cache_data
def load_full_dataset():
    try:
        df = pd.read_csv("new_df.csv")
        
        # Basic data validation and cleaning
        df.fillna("Unknown", inplace=True)
        
        # Create enhanced search text
        df["search_text"] = df.apply(lambda row: 
            f"Date: {row.get('date', 'Unknown')} | "
            f"State: {row.get('state', 'Unknown')} | "
            f"Shape: {row.get('shape', 'Unknown')} | "
            f"Duration: {row.get('duration', 'Unknown')} | "
            f"Comments: {row.get('comments', 'No comments')}", axis=1)
            
        return df
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return None

df = load_full_dataset()

# Full dataset analysis function
def analyze_full_dataset(question, df):
    if not st.session_state.client:
        return "API client not initialized", None
    
    # Prepare the full dataset for analysis
    full_context = f"COMPLETE UFO SIGHTINGS DATASET SUMMARY:\n"
    full_context += f"Total sightings: {len(df)}\n"
    
    # Add key statistics
    if 'date' in df.columns:
        full_context += f"Date range: {df['date'].min()} to {df['date'].max()}\n"
    if 'state' in df.columns:
        top_locations = df['state'].value_counts().head(5).to_dict()
        full_context += f"Top States: {', '.join(top_locations.keys())}\n"
    if 'shape' in df.columns:
        top_shapes = df['shape'].value_counts().head(5).to_dict()
        full_context += f"Top shapes: {', '.join(top_shapes.keys())}\n"
    
    # Sample some representative entries
    sample_size = min(10, len(df))
    sample = df.sample(sample_size)
    full_context += f"\nSAMPLE SIGHTINGS ({sample_size} of {len(df)}):\n"
    for idx, row in sample.iterrows():
        full_context += f"\n- {row['search_text']}\n"
    
    # Generate comprehensive answer
    try:
        response = st.session_state.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior UFO data analyst. Provide comprehensive answers "
                    "based on the full dataset. Identify patterns, trends, and notable findings."
                },
                {
                    "role": "user",
                    "content": f"QUESTION: {question}\n\n"
                    f"DATASET OVERVIEW:\n{full_context}\n\n"
                    "ANALYSIS INSTRUCTIONS:\n"
                    "1. Consider ALL sightings in your analysis\n"
                    "2. Identify statistical patterns where possible\n"
                    "3. Note data limitations or biases\n"
                    "4. Provide specific examples when relevant\n"
                    "5. Include quantitative insights if available\n\n"
                    "COMPREHENSIVE ANSWER:"
                }
            ],
            temperature=0.1,  # Lower temp for more factual responses
            max_tokens=500
        )
        
        # Also get the most relevant individual sightings
        if len(df) > 0:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf = vectorizer.fit_transform(df["search_text"])
            question_vec = vectorizer.transform([question])
            df['similarity'] = cosine_similarity(question_vec, tfidf).flatten()
            top_sightings = df.sort_values('similarity', ascending=False).head(5)
        else:
            top_sightings = pd.DataFrame()
            
        return response.choices[0].message.content, top_sightings
    except Exception as e:
        return f"Error generating answer: {str(e)}", None

# Main interface
st.title("🛸 UFO Sightings Database Enquiry")
st.write("Get comprehensive answers based on 1990 to 2013 UFO Sightings in US")

question = st.text_area(
    "Ask your question:", 
    placeholder="e.g. Which US State has the most UFO Sightings?",
    height=100
)

if st.button("Analyze Dataset") and question:
    if not st.session_state.client:
        st.warning("Please enter your OpenAI API key")
    elif df is None:
        st.error("Data not loaded properly")
    else:
        with st.spinner("Analyzing dataset..."):
            # Get analysis of full dataset
            answer, top_sightings = analyze_full_dataset(question, df)
            
            # Display results
            st.subheader("Comprehensive Analysis")
            st.markdown(answer)
            
            if top_sightings is not None and not top_sightings.empty:
                st.subheader("Related Individual Sightings")
                st.dataframe(
                    top_sightings.drop(columns=['search_text', 'similarity'], errors='ignore'),
                    hide_index=True,
                    use_container_width=True
                )
            
            # Show dataset stats
            st.subheader("Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sightings", len(df))
            with col2:
                if 'date' in df.columns:
                    st.metric("Date Range", f"{df['date'].min()} to {df['date'].max()}")
            with col3:
                if 'state' in df.columns:
                    st.metric("Top States", df['state'].mode()[0] if len(df) > 0 else "N/A")

# Add data export option
if df is not None:
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label="Download Dataset",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="ufo_sightings.csv",
        mime="text/csv"
    )