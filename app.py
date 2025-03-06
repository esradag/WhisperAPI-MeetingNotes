import streamlit as st
import tempfile
import os
import re
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# LangChain and OpenAI imports
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
import openai

# Load .env file if it exists
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Voice-to-Meeting Notes Converter",
    page_icon="🎙️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 4px solid #1E88E5;
    }
    .action-item {
        background-color: #FFECB3;
        padding: 0.6rem;
        border-radius: 0.3rem;
        border-left: 4px solid #FFA000;
        margin-bottom: 0.5rem;
    }
    .key-point {
        background-color: #E8F5E9;
        padding: 0.6rem;
        border-radius: 0.3rem;
        border-left: 4px solid #43A047;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<div class='main-header'>Voice-to-Meeting Notes Converter</div>", unsafe_allow_html=True)
st.markdown("""
This application uses OpenAI's Whisper API to transcribe meeting recordings and
then extracts key points and actionable items using LangChain. Upload an audio recording to get started.
""")

# API Key input
api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
if api_key:
    openai.api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key

# Function to transcribe audio using Whisper API
def transcribe_audio(audio_file):
    try:
        with open(audio_file, "rb") as file:
            transcription = openai.Audio.transcribe(
                model="whisper-1",
                file=file
            )
        return transcription["text"]
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

# Function to create and run LangChain chains for meeting analysis
def analyze_transcript_with_langchain(transcript):
    try:
        # Initialize the language model
        llm = ChatOpenAI(model_name="gpt-4", temperature=0.3)
        
        # Set up text splitter for long transcripts
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
        
        # Process transcript based on length
        if len(transcript) > 4000:
            chunks = text_splitter.split_text(transcript)
            # For simplicity, we're using the first chunk
            # In a production app, you might want to process all chunks or use a map-reduce approach
            transcript_to_process = chunks[0]
            st.info("The transcript is long, analyzing the first part only. Full transcript is still available.")
        else:
            transcript_to_process = transcript
        
        # Define prompt templates
        summary_prompt = PromptTemplate(
            input_variables=["transcript"],
            template="You are an expert meeting assistant. Provide a concise summary (2-3 paragraphs) of this meeting transcript.\n\n{transcript}"
        )
        
        key_points_prompt = PromptTemplate(
            input_variables=["transcript", "summary"],
            template="You are an expert meeting assistant. Extract the most important key points from this meeting transcript. Use this summary as reference: {summary}\n\nTranscript: {transcript}\n\nFormat them as a numbered list with clear, concise points."
        )
        
        action_items_prompt = PromptTemplate(
            input_variables=["transcript", "key_points"],
            template="You are an expert meeting assistant. Extract all action items from this meeting transcript. The key points are: {key_points}\n\nTranscript: {transcript}\n\nFormat them as a numbered list with the following for each item: 1) The action to be done, 2) Who is responsible (if mentioned), 3) Any deadline mentioned. If some information is not available, just include what is known."
        )
        
        # Create individual chains
        summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")
        key_points_chain = LLMChain(llm=llm, prompt=key_points_prompt, output_key="key_points")
        action_items_chain = LLMChain(llm=llm, prompt=action_items_prompt, output_key="action_items")
        
        # Create sequential chain
        sequential_chain = SequentialChain(
            chains=[summary_chain, key_points_chain, action_items_chain],
            input_variables=["transcript"],
            output_variables=["summary", "key_points", "action_items"],
            verbose=True
        )
        
        # Run the chain
        result = sequential_chain({"transcript": transcript_to_process})
        
        return result
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return None

# Function to parse action items into a dataframe
def parse_action_items(action_items_text):
    # This regex pattern looks for numbered items possibly containing "who:" and "when:" or similar patterns
    action_items_list = re.findall(r'\d+\.\s*(.*?)(?=\d+\.|$)', action_items_text, re.DOTALL)
    
    actions = []
    owners = []
    deadlines = []
    
    for item in action_items_list:
        item = item.strip()
        
        # Try to extract owner (who)
        owner_match = re.search(r'(?:who|responsible|assigned to|owner):\s*(.*?)(?:\.|\n|$|when|deadline|by)', item, re.IGNORECASE)
        owner = owner_match.group(1).strip() if owner_match else ""
        
        # Try to extract deadline
        deadline_match = re.search(r'(?:when|deadline|by|due|complete by):\s*(.*?)(?:\.|\n|$)', item, re.IGNORECASE)
        deadline = deadline_match.group(1).strip() if deadline_match else ""
        
        # Clean up the action text by removing the extracted parts
        action_text = item
        if owner_match:
            action_text = re.sub(r'(?:who|responsible|assigned to|owner):\s*.*?(?:\.|\n|$|when|deadline|by)', '', action_text, flags=re.IGNORECASE)
        if deadline_match:
            action_text = re.sub(r'(?:when|deadline|by|due|complete by):\s*.*?(?:\.|\n|$)', '', action_text, flags=re.IGNORECASE)
        
        # Further cleanup
        action_text = re.sub(r'^\s*-\s*', '', action_text).strip()
        
        actions.append(action_text)
        owners.append(owner)
        deadlines.append(deadline)
    
    return pd.DataFrame({
        "Action": actions,
        "Owner": owners,
        "Deadline": deadlines,
        "Status": ["Pending"] * len(actions)
    })

# Function to format transcript with timestamps
def format_transcript(transcript):
    # In a real application, if whisper provides timestamps, you could use those
    # For now, we'll just return the transcript as is
    return transcript

# File uploader
uploaded_file = st.file_uploader("Upload Meeting Audio", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    with st.spinner("Processing audio file..."):
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix="."+uploaded_file.name.split(".")[-1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        # Check if API key is provided
        if not api_key:
            st.warning("Please enter your OpenAI API key in the sidebar.")
        else:
            # Transcribe the audio
            transcript = transcribe_audio(temp_file_path)
            
            # Remove the temporary file
            os.unlink(temp_file_path)
            
            if transcript:
                # Create tabs for different views
                transcript_tab, summary_tab, key_points_tab, action_items_tab, export_tab = st.tabs([
                    "📝 Transcript", "📋 Summary", "🔑 Key Points", "✅ Action Items", "📤 Export"
                ])
                
                # Analyze the transcript using LangChain
                analysis_results = analyze_transcript_with_langchain(transcript)
                
                if analysis_results:
                    # Transcript tab
                    with transcript_tab:
                        st.markdown("<div class='section-header'>Meeting Transcript</div>", unsafe_allow_html=True)
                        st.text_area("Full Transcript", format_transcript(transcript), height=400)
                    
                    # Summary tab
                    with summary_tab:
                        st.markdown("<div class='section-header'>Meeting Summary</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='highlight'>{analysis_results['summary']}</div>", unsafe_allow_html=True)
                    
                    # Key points tab
                    with key_points_tab:
                        st.markdown("<div class='section-header'>Key Points</div>", unsafe_allow_html=True)
                        
                        # Process the key points to display with proper formatting
                        key_points = analysis_results["key_points"].split("\n")
                        for point in key_points:
                            if point.strip():
                                st.markdown(f"<div class='key-point'>{point}</div>", unsafe_allow_html=True)
                    
                    # Action items tab
                    with action_items_tab:
                        st.markdown("<div class='section-header'>Action Items</div>", unsafe_allow_html=True)
                        
                        # Display action items as a dataframe
                        action_items_df = parse_action_items(analysis_results["action_items"])
                        edited_df = st.data_editor(action_items_df, use_container_width=True)
                        
                        # Process the action items to display with proper formatting
                        action_items = analysis_results["action_items"].split("\n")
                        for item in action_items:
                            if item.strip():
                                st.markdown(f"<div class='action-item'>{item}</div>", unsafe_allow_html=True)
                    
                    # Export tab
                    with export_tab:
                        st.markdown("<div class='section-header'>Export Meeting Notes</div>", unsafe_allow_html=True)
                        
                        # Generate export text
                        meeting_date = datetime.now().strftime("%Y-%m-%d")
                        export_text = f"""# Meeting Notes - {meeting_date}

## Summary
{analysis_results['summary']}

## Key Points
{analysis_results['key_points']}

## Action Items
{analysis_results['action_items']}

## Full Transcript
{transcript}
"""
                        
                        # Display export options
                        st.download_button(
                            label="Download as Markdown",
                            data=export_text,
                            file_name=f"meeting_notes_{meeting_date}.md",
                            mime="text/markdown"
                        )
                        
                        # Also provide CSV export for action items
                        csv = action_items_df.to_csv(index=False)
                        st.download_button(
                            label="Download Action Items as CSV",
                            data=csv,
                            file_name=f"action_items_{meeting_date}.csv",
                            mime="text/csv"
                        )

# Instructions in the sidebar
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("""
1. Enter your OpenAI API key above
2. Upload an audio file of your meeting
3. Wait for processing to complete
4. View the transcription and analysis
5. Export your meeting notes
""")

# About section in the sidebar
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This application uses:
- OpenAI's Whisper API for speech-to-text
- LangChain for orchestrating the analysis pipeline
- OpenAI's GPT-4 for analysis and summarization
- Streamlit for the web interface

The app extracts:
- Complete meeting transcript
- Key discussion points
- Action items with owners and deadlines
- Meeting summary
""")

if __name__ == "__main__":
    # Main entry point of the Streamlit application
    pass