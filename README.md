# 🎙️ Voice-to-Meeting Notes Converter

A Streamlit application that automatically transcribes meeting audio recordings, extracts important points, and generates actionable items using LangChain and OpenAI.

## 📋 Features

- **Audio Transcription**: Convert meeting recordings to text using OpenAI Whisper API
- **Intelligent Analysis**: Extract key points and action items from transcripts using LangChain and GPT-4
- **Meeting Summary**: Generate a concise and clear summary of the meeting
- **Action Items**: Create a structured list of tasks with assignees and deadlines
- **Easy Export**: Save meeting notes in Markdown or CSV format

## 🚀 Installation

### Option 1: Standard Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/voice-to-meeting-notes.git
cd voice-to-meeting-notes

# Install required libraries
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Option 2: Using Python 3.11 (Recommended)
For better compatibility with dependencies, we recommend using Python 3.11:

```bash
# Clone the repository
git clone https://github.com/yourusername/voice-to-meeting-notes.git
cd voice-to-meeting-notes

# Create a Python 3.11 virtual environment
python3.11 -m venv venv_py311
source venv_py311/bin/activate

# Install required libraries
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 📝 Usage

1. Enter your OpenAI API key in the sidebar
2. Upload your meeting audio recording (MP3, WAV, or M4A format)
3. After the transcription and analysis is complete:
   - View the full transcript
   - Review the meeting summary
   - Check the key points
   - Edit action items
4. Export the results in Markdown or CSV format

## 🛠️ Technologies

- [Streamlit](https://streamlit.io/) - Web interface
- [OpenAI Whisper API](https://openai.com/research/whisper) - Speech-to-text conversion
- [LangChain](https://github.com/hwchase17/langchain) - LLM orchestration framework
- [OpenAI GPT-4](https://openai.com/gpt-4) - Text analysis and summarization
- [Pandas](https://pandas.pydata.org/) - Data processing

## 🔄 LangChain Implementation

This application leverages LangChain to:
- Create a sequential processing pipeline for meeting analysis
- Handle large transcripts through text splitting and chunking
- Maintain context between summary, key points, and action items extraction
- Provide a modular architecture for easy extension

## 📸 Screenshots

![App Interface](screenshots/app_interface.png)
*Main application interface*

![Action Items](screenshots/action_items.png)
*Action items table*

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please open an issue before sending a pull request.

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request