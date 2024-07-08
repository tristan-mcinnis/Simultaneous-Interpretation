import os
import time
import re
import wave
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import pyaudio
import traceback
import threading
import queue
from openai import OpenAI

# Set the environment variable to allow duplicate OpenMP runtime initialization
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from faster_whisper import WhisperModel
import speech_recognition as sr
from rich.console import Console
from rich.panel import Panel

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from the .env file
api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is loaded correctly
if api_key is None:
    raise ValueError("API key not found. Please make sure it's set in the .env file.")

openai_client = OpenAI(api_key=api_key)

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Define the translation template
translation_template = """
Translate the following sentence into {language}, return ONLY the translation, nothing else.

Sentence: {sentence}

Previous Chunks: {previous_chunks}

Topic: {topic}
"""

output_parser = StrOutputParser()
llm = ChatOpenAI(temperature=0.0, model="gpt-4o", api_key=api_key)
translation_prompt = ChatPromptTemplate.from_template(translation_template)

translation_chain = (
    {"language": RunnablePassthrough(), "sentence": RunnablePassthrough(), "previous_chunks": RunnablePassthrough(), "topic": RunnablePassthrough()} 
    | translation_prompt
    | llm
    | output_parser
)

def load_dictionary(dictionary_path):
    custom_dict = {}
    with open(dictionary_path, 'r', encoding='utf-8') as file:
        for line in file:
            if '=' in line:
                term, translation = line.strip().split('=', 1)
                custom_dict[term] = translation
            else:
                log(f"Skipping invalid dictionary line: {line.strip()}")
    return custom_dict

def preprocess_text(text, custom_dict):
    for term, translation in custom_dict.items():
        text = text.replace(term, translation)
    return text

def translate(sentence, language="French", previous_chunks="", topic=""):
    data_input = {"language": language, "sentence": sentence, "previous_chunks": previous_chunks, "topic": topic}
    translation = translation_chain.invoke(data_input)
    return translation

# Initialize Rich Console
console = Console()

# Initialize Whisper model
num_cores = os.cpu_count()
whisper_model = WhisperModel('medium', device='cpu', compute_type='int8', cpu_threads=num_cores // 2, num_workers=num_cores // 2)

r = sr.Recognizer()

# To capture console output
captured_output = []

# Initialize queues for transcription results, translations, and TTS segments
transcription_queue = queue.Queue()
translation_queue = queue.Queue()
tts_queue = queue.Queue()

previous_chunks = []

def log(message):
    if isinstance(message, Panel):
        console.print(message)
    else:
        console.print(message)
        captured_output.append(message)
    with open("logfile.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"{datetime.now()} - {message}\n")

def wav_to_text(audio_path, language):
    segments, _ = whisper_model.transcribe(audio_path, language=language)
    text = ''.join(segment.text for segment in segments)
    return text

def text_to_speech(text, output_device_index):
    try:
        player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True, output_device_index=output_device_index)
        stream_start = False
        with openai_client.audio.speech.with_streaming_response.create(
            model='tts-1', voice='nova', response_format='pcm', speed='1.25', input=text
        ) as response:
            silence_threshold = 0.01
            for chunk in response.iter_bytes(chunk_size=1024):
                if stream_start:
                    player_stream.write(chunk)
                else:
                    if max(chunk) > silence_threshold:
                        player_stream.write(chunk)
                        stream_start = True
    except Exception as e:
        log(f"Error in TTS: {e}")
        log(traceback.format_exc())

def callback(recognizer, audio, language, translate_to, output_device_index, audio_translate, custom_dict, topic):
    try:
        prompt_audio_path = 'prompt.wav'
        with open(prompt_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())
        prompt_text = wav_to_text(prompt_audio_path, language)
        prompt_text = preprocess_text(prompt_text, custom_dict)  # Preprocess the text
        transcription_queue.put(prompt_text)
    except Exception as e:
        log(f"Error in callback: {e}")
        log(traceback.format_exc())

def transcription_worker():
    while True:
        try:
            text = transcription_queue.get()
            if text is None:
                break
            log(text)
            translation_queue.put(text)
            transcription_queue.task_done()
        except Exception as e:
            log(f"Error in transcription_worker: {e}")
            log(traceback.format_exc())

def translation_worker(language, audio_translate, topic):
    global previous_chunks
    max_chunks = 10  # Increase the number of previous chunks considered
    while True:
        try:
            text = translation_queue.get()
            if text is None:
                break
            translated_text = translate(text, language, "\n".join(previous_chunks[-max_chunks:]), topic)
            log(f"Translated: {translated_text}")
            previous_chunks.append(translated_text)
            if len(previous_chunks) > max_chunks:
                previous_chunks.pop(0)
            if audio_translate:
                tts_queue.put(translated_text)
            translation_queue.task_done()
        except Exception as e:
            log(f"Error in translation_worker: {e}")
            log(traceback.format_exc())

def tts_worker(output_device_index):
    while True:
        try:
            text = tts_queue.get()
            if text is None:
                break
            text_to_speech(text, output_device_index)
            tts_queue.task_done()
        except Exception as e:
            log(f"Error in tts_worker: {e}")
            log(traceback.format_exc())

def save_log_to_file(text):
    downloads_folder = str(Path.home() / "Downloads")
    filename = datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt"
    file_path = os.path.join(downloads_folder, filename)
    
    # Using regex to format the text
    transcript_lines = []
    translation_lines = []
    for line in text.splitlines():
        if line.startswith('Translated:'):
            translation_lines.append(line.replace('Translated: ', '').strip())
        else:
            transcript_lines.append(line.strip())
    
    formatted_text = "Transcript:\n" + "\n".join(transcript_lines) + "\n\nTranslated:\n" + "\n".join(translation_lines)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(formatted_text)
    log(Panel(f"Transcription saved to {file_path}", border_style="bold green", title="LOG"))

def start_listening(language='en', input_device_index=0, translate_to='French', output_device_index=0, audio_translate=False, custom_dict=None, topic=""):
    log(Panel(f"Adjusting for ambient noise... (Language: {language})", border_style="blue1", title="ACTION"))
    source = sr.Microphone(device_index=input_device_index)
    with source:
        r.adjust_for_ambient_noise(source, duration=2)
    
    log(Panel("Start speaking / start audio. Press 'CTRL + C' to exit", border_style="green1", title="ACTION"))
    stop_listening = r.listen_in_background(source, lambda recognizer, audio: callback(recognizer, audio, language, translate_to, output_device_index, audio_translate, custom_dict, topic), phrase_time_limit=8)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_listening(wait_for_stop=False)
        transcription_queue.put(None)  # Signal the workers to exit
        translation_queue.put(None)
        tts_queue.put(None)
        transcription_thread.join()  # Wait for the workers to finish
        translation_thread.join()
        tts_thread.join()
        combined_text = '\n'.join(str(msg) for msg in captured_output if not isinstance(msg, Panel))
        save_log_to_file(combined_text)
        log(Panel("Listening stopped.", border_style="magenta3", title="ACTION"))

def list_audio_devices():
    audio = pyaudio.PyAudio()
    input_devices = []
    output_devices = []

    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        device_name = device_info.get('name')
        max_input_channels = device_info.get('maxInputChannels')
        max_output_channels = device_info.get('maxOutputChannels')

        if max_input_channels > 0:
            input_devices.append((i, device_name, max_input_channels))
        elif max_output_channels > 0:
            output_devices.append((i, device_name, max_output_channels))

    console.print("#### INPUT CHANNELS")
    for index, name, channels in input_devices:
        console.print(f"{index}: {name} (Max Input Channels: {channels})")

    console.print("\n#### OUTPUT CHANNELS")
    for index, name, channels in output_devices:
        console.print(f"{index}: {name} (Max Output Channels: {channels})")

    audio.terminate()

if __name__ == "__main__":
    try:
        list_audio_devices()
        input_device_index = int(input("Enter the device index for the microphone you want to use: ").strip())
        output_device_index = int(input("Enter the device index for the speaker you want to use: ").strip())
        language_code = input("Enter input language code (zh or en): ").strip().lower()

        if language_code not in {"zh", "en"}:
            log("Invalid language code. Use 'zh' for Chinese or 'en' for English.")
        else:
            translation_option = input("Do you want to translate the transcription (yes or no)? ").strip().lower()
            translate_option = translation_option == "yes"
            
            audio_translation_option = input("Do you want audio translation (yes or no)? ").strip().lower()
            audio_translate = audio_translation_option == "yes"
            
            custom_dict_path = input("Enter the path to your custom dictionary file (or leave empty to skip): ").strip()
            custom_dict = load_dictionary(custom_dict_path) if custom_dict_path else None

            topic = input("Enter the topic of the conversation: ").strip()

            # Start worker threads
            transcription_thread = threading.Thread(target=transcription_worker)
            translation_thread = threading.Thread(target=translation_worker, args=(translate_option, audio_translate, topic))
            tts_thread = threading.Thread(target=tts_worker, args=(output_device_index,))
            
            transcription_thread.start()
            translation_thread.start()
            tts_thread.start()

            start_listening(language_code, input_device_index, translate_option, output_device_index, audio_translate, custom_dict, topic)
            
    except Exception as e:
        log(f"Error in main: {e}")
        log(traceback.format_exc())
