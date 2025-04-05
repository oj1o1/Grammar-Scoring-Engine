import streamlit as st
import assemblyai as aai
import os
import re
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# Configure Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("⚠️ Gemini API key not found. Please set it in your .env file.")

# Configure AssemblyAI
if ASSEMBLYAI_API_KEY:
    aai.settings.api_key = ASSEMBLYAI_API_KEY
else:
    st.error("⚠️ AssemblyAI API key not found. Please set it in your .env file.")

# Initialize session state
if "errors" not in st.session_state:
    st.session_state.errors = []

# Title
st.title("📝 Grammar Scoring Engine")
st.subheader("Record your voice to receive instant grammar and sentiment feedback")

# Tabs
tab1, tab2 = st.tabs(["🎙️ Record Audio", "📂 Upload Audio"])
audio_file = None

# 🎙️ Record Audio
with tab1:
    st.write("Click to record your voice sample:")
    audio_file = st.audio_input("🎤 Start Recording", key="audio_input")
    if audio_file:
        st.success("✅ Recording complete!")
        st.audio(audio_file)

# 📂 Upload Audio
with tab2:
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
    if uploaded_file:
        audio_file = uploaded_file
        st.audio(audio_file)

# 🔍 Process Audio
if audio_file:
    with st.spinner("⏳ Processing audio..."):
        try:
            # Save audio file temporarily
            with open("temp_audio.wav", "wb") as f:
                f.write(audio_file.getvalue())

            # Transcribe audio using AssemblyAI
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe("temp_audio.wav")
            transcription = transcript.text.strip()

            st.markdown("### 📝 Transcription:")
            if transcription:
                st.write(transcription)

                if GEMINI_API_KEY:
                    try:
                        # Initialize Gemini model
                        gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash-lite")

                        # GRAMMAR FEEDBACK
                        grammar_prompt = f"""
Analyze the following sentence for grammatical correctness. Identify any grammar issues, provide a corrected version, give a grammar score out of 100, and explain the reasoning.

Sentence: "{transcription}"

Format your response strictly as:
Grammatical Issues: <issues here>
Corrected Version: <corrected sentence>
Grammar Score: <numeric score>/100
Explanation: <brief explanation>
"""
                        grammar_response = gemini_model.generate_content(grammar_prompt)
                        grammar_text = grammar_response.text

                        st.markdown("### 🤖 Grammar Feedback:")
                        st.write(grammar_text)

                        # Extract score
                        match = re.search(r"Grammar Score:\s*(\d{1,3})\s*/\s*100", grammar_text, re.IGNORECASE)
                        grammar_score = int(match.group(1)) if match else 100
                        grammar_score = min(max(grammar_score, 0), 100)
                        st.markdown("### 📈 Grammar Score:")
                        st.progress(grammar_score / 100)
                        st.write(f"✅ Your grammar score is: **{grammar_score}/100**")

                        # Extract corrected version
                        corrected_match = re.search(r"Corrected Version:\s*(.+)", grammar_text)
                        if corrected_match:
                            corrected_text = corrected_match.group(1).strip()
                            st.download_button("📥 Download Corrected Transcript", corrected_text, file_name="corrected_transcript.txt")

                        # SENTIMENT ANALYSIS
                        sentiment_prompt = f"""
Analyze the sentiment of the following sentence and classify it as Positive, Negative, or Neutral. Provide a short explanation for your classification.

Sentence: "{transcription}"

Format your response strictly as:
Sentiment: <Positive/Negative/Neutral>
Explanation: <brief explanation>
"""
                        sentiment_response = gemini_model.generate_content(sentiment_prompt)
                        sentiment_text = sentiment_response.text

                        st.markdown("### 💬 Sentiment Analysis:")
                        st.write(sentiment_text)

                    except Exception as e:
                        error_msg = f"⚠️ Error with Gemini API: {e}"
                        st.session_state.errors.append(error_msg)
                        st.error(error_msg)
            else:
                st.warning("⚠️ No transcription detected. Please try again.")

        except Exception as e:
            error_msg = f"⚠️ Error processing audio: {e}"
            st.session_state.errors.append(error_msg)
            st.error(error_msg)

# 🐞 Error Log
if st.session_state.errors:
    with st.expander("🐞 Error Log"):
        for err in st.session_state.errors:
            st.code(err)
