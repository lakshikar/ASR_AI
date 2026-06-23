import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
from pydub import AudioSegment
# Import necessary libraries for the actual transcription step
# If you use OpenAI's API, uncomment the following:
# from openai import OpenAI 

import torch
from transformers import pipeline
import librosa
import requests
from transformers import pipeline
import scipy.io.wavfile
from datasets import load_dataset
import nltk # Standard text chunking helper


# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt', quiet=True)
# Ensure both punkt and the required punkt_tab are pre-downloaded automatically
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

try:
    # 1. Initialize the pipeline
    tts_pipeline = pipeline("text-to-speech", model="microsoft/speecht5_tts", device=0 if torch.cuda.is_available() else -1)
    
    # FIX: Point directly to an auto-converted Parquet mirror to avoid script errors completely
    embeddings_dataset = load_dataset("regisss/cmu-arctic-xvectors", split="validation")
    
    # Choose your default voice index 
    raw_embedding = embeddings_dataset[7306]["xvector"]
    DEFAULT_SPEAKER_EMBEDDING = torch.tensor(raw_embedding).unsqueeze(0)
    
    TTS_AVAILABLE = True
    print("✅ TTS Pipeline and voice profiles initialized successfully.")
except Exception as e:
    print(f"Warning: Could not initialize TTS pipeline. Text-to-Speech feature will be disabled. Error: {e}")
    TTS_AVAILABLE = False



# --- Configuration ---
SAMPLE_RATE = 44100  # Standard sample rate for audio recording
DURATION = 5         # Duration to record in seconds
OUTPUT_FILENAME = "recorded_audio.wav"
TRANSCRIPTION_MODEL = "openai/whisper-base" # Placeholder for the model name


# --- Step 1: Audio Recording Function ---

def record_audio(duration: int, sample_rate: int, filename: str) -> bool:
    """
    Records audio from the default microphone for a specified duration.
    Saves the raw audio data as a WAV file.
    """
    print("="*50)
    print(f"🎙️ Starting audio recording...")
    print(f"🎤 Please speak clearly for {duration} seconds.")
    print("="*50)
    
    try:
        # Record audio data
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished
        
        # Save the recording to a WAV file
        write(filename, sample_rate, recording)
        print(f"\n✅ Recording complete! Audio saved successfully to '{filename}'")
        return True
    except Exception as e:
        print(f"\n❌ Error during recording: {e}")
        print("Please ensure your microphone is connected and accessible.")
        return False


# --- Configuration ---
# Use a specific model name for Whisper, or let the pipeline choose a default.
# 'openai/whisper-base' is a good starting point.
MODEL_NAME = "openai/whisper-base" 

def transcribe_audio_hf(audio_file_path: str, model_name: str = MODEL_NAME) -> str:
    """
    Transcribes an audio file using the Hugging Face transformers pipeline (Whisper).

    Args:
        audio_file_path: Path to the audio file (e.g., .wav, .mp3).
        model_name: The pre-trained model to use.

    Returns:
        The transcribed text string.
    """
    print(f"--- Starting Transcription using Hugging Face Model: {model_name} ---")
    
    # 1. Initialize the pipeline
    # We specify the task as 'automatic-speech-recognition'
    try:
        asr_pipeline = pipeline(
            "automatic-speech-recognition", 
            model=model_name, 
            device=0 if torch.cuda.is_available() else -1 # Use GPU if available
        )
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("Please ensure you have 'transformers', 'accelerate', and 'torch' installed.")
        return ""

    # 2. Load the audio file using librosa
    try:
        # librosa loads audio and resamples it to a standard format (e.g., 16kHz)
        # The pipeline often expects the audio to be a dictionary containing the array.
        audio_input, sr = librosa.load(audio_file_path, sr=16000)
        print(f"Audio loaded successfully. Sample Rate: {sr} Hz, Duration: {len(audio_input)/sr:.2f} seconds.")
    except FileNotFoundError:
        print(f"Error: Audio file not found at {audio_file_path}")
        return ""
    except Exception as e:
        print(f"Error loading audio with librosa: {e}")
        return ""

    # 3. Transcribe the audio
    try:
        # The pipeline expects the audio data in a specific format.
        result = asr_pipeline(
            audio_input, 
            chunk_length_s=30, # Process in chunks if the audio is very long
            return_timestamps=False
        )
        
        transcribed_text = result["text"]
        print("--- Transcription Complete ---")
        return transcribed_text

    except Exception as e:
        print(f"An error occurred during the pipeline execution: {e}")
        return ""
    

def call_ollama(transcript: str) -> str:
    """Call Ollama using requests library"""
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llama3.2:latest",
            "prompt": f"You are a helpful assistant. : {transcript}",
            "stream": False
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get('response', '')
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return ""    

# --- New Function ---
# def text_to_speech(text: str, output_filename: str = "output_audio.wav"):
#     """
#     Converts the given text string into an audio file using a pre-trained TTS model.
    
#     Args:
#         text: The text to synthesize.
#         output_filename: The name of the WAV file to save the audio to.
#     """
#     if not TTS_AVAILABLE:
#         print("TTS feature is unavailable. Skipping audio generation.")
#         return None

#     print(f"\n🔊 Generating speech for: '{text[:50]}...'")
#     try:
#         # FIX: Added forward_params so the speaker embedding is properly sent to SpeechT5
#         tts_output = tts_pipeline(text, forward_params={"speaker_embeddings": DEFAULT_SPEAKER_EMBEDDING})
        
#         audio_array = tts_output["audio"]
#         sampling_rate = tts_output["sampling_rate"]
        
#         # FIX: Scaled the raw float array safely to 16-bit PCM integers
#         audio_data = (audio_array * 32767).astype(np.int16)
#         scipy.io.wavfile.write(output_filename, rate=sampling_rate, data=audio_data)
        
#         print(f"✅ Successfully saved audio to {output_filename}")
#         return output_filename
#     except Exception as e:
#         print(f"❌ Error during TTS generation: {e}")
#         return None

# ======================================================================
# --- EXAMPLE USAGE ---
# ======================================================================


def text_to_speech(text: str, output_filename: str = "output_audio.wav"):
    """
    Converts the given text string into an audio file using a pre-trained TTS model.
    Handles long text strings by breaking them down into sentences.
    """
    if not TTS_AVAILABLE:
        print("TTS feature is unavailable. Skipping audio generation.")
        return None

    print(f"\n🔊 Generating speech for: '{text[:50]}...'")
    try:
        # Use nltk to cleanly split the long text response into individual sentences
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        
        combined_audio = []
        sampling_rate = 16000 # SpeechT5 standard output rate

        for sentence in sentences:
            # Skip empty entries
            if not sentence.strip():
                continue
                
            # Synthesize each sentence independently to avoid the 600-token limit
            tts_output = tts_pipeline(sentence, forward_params={"speaker_embeddings": DEFAULT_SPEAKER_EMBEDDING})
            audio_array = tts_output["audio"]
            sampling_rate = tts_output["sampling_rate"]
            
            # Append this chunk's raw audio array
            combined_audio.append(audio_array)
            
            # Optional: Add a brief 0.2 second pause (silence array) between sentences
            pause_samples = int(sampling_rate * 0.2)
            combined_audio.append(np.zeros(pause_samples))

        if not combined_audio:
            return None

        # Concatenate all individual sentence audio blocks into a single piece
        final_audio_array = np.concatenate(combined_audio)
        
        # Scale float audio safely to 16-bit PCM integers to prevent static noise
        audio_data = (final_audio_array * 32767).astype(np.int16)
        scipy.io.wavfile.write(output_filename, rate=sampling_rate, data=audio_data)
        
        print(f"✅ Successfully saved chunked audio to {output_filename}")
        return output_filename
        
    except Exception as e:
        print(f"❌ Error during TTS generation: {e}")
        return None





# IMPORTANT: Replace 'path/to/your/audio.wav' with an actual path to an audio file.
# For best results with Whisper, use WAV files sampled at 16kHz.
AUDIO_FILE_PATH = "recorded_audio.wav" 
def main():
    """
    Orchestrates the recording and transcription process.
    """
    # 1. Record the audio
    if not record_audio(DURATION, SAMPLE_RATE, OUTPUT_FILENAME):
        print("\nProcess aborted due to recording failure.")
        return

    # 2. Transcribe the audio
    transcript = transcribe_audio_hf(OUTPUT_FILENAME, TRANSCRIPTION_MODEL)

    # 3. Display Results
    print("\n" + "="*50)
    print("✨ TRANSCRIPTION COMPLETE ✨")
    print("="*50)
    print(f"🎤 Original Audio File: {OUTPUT_FILENAME}")
    print("\n📝 Transcript:")
    print(transcript)
    print("="*50)
    response = call_ollama(transcript)
    print("\n🤖 Ollama Response:")
    print(response)
    print("="*50)

    # 4. Generate Speech from the Response
    if TTS_AVAILABLE:
        audio_filename = text_to_speech(response)
        if audio_filename:
            # FIX: Changed 'afplay' (macOS) to 'paplay' (Ubuntu Linux)
            os.system(f"paplay {audio_filename}")
        else:
            print("Audio generation failed. Skipping audio playback.")
    else:
        print("TTS feature is unavailable. Skipping audio generation and playback.")

    # 4. Cleanup (Optional)
    # os.remove(OUTPUT_FILENAME)
    # print(f"\n🧹 Cleaned up temporary file: {OUTPUT_FILENAME}")


if __name__ == "__main__":
    main()