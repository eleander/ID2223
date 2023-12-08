from transformers import pipeline
import gradio as gr
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

pipe = pipeline(model="Potatoasdasdasdasda/whisper-base-es-improved-2") 

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

def create_chat_history(chat_history, msg):
    messages = [{"role": "system", "content": "Tu eres un asistente util."}]
    for request, response in chat_history:
        messages.append({"role": "user", "content": request})
        messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": msg})
    return messages

def respond(audio, chat_history):
    bot_request = transcribe(audio)
    
    bot_response = client.chat.completions.create(
        messages=create_chat_history(chat_history, bot_request),
        model="gpt-3.5-turbo",
    ).choices[0].message.content
        
    chat_history.append((bot_request, bot_response))
    return None, chat_history

with gr.Blocks() as demo:
    with gr.Tab("Microphone Mode"):
        iface = gr.Interface(
            fn=transcribe, 
            inputs=gr.Audio(sources=["microphone", "upload"], type="filepath"),
            outputs="text",
            title="Whisper Base Spanish Improved",
            description="Realtime demo for Spanish speech recognition using a fine-tuned Whisper Base model.",
        )

    with gr.Tab("Conversation Mode"):
        chatbot = gr.Chatbot(show_copy_button=True)
        mic = gr.Audio(sources="microphone", type="filepath")
        mic.stop_recording(respond, [mic, chatbot], [mic,chatbot])
        
demo.launch()