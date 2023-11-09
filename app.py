import os
import secrets
from typing import Optional

import gradio as gr
import openai
from openai import OpenAI
from config import *

user_id_length = 13
openai.api_key = OPENAI_KEY

from TTS.api import TTS
#import pyttsx3
model_path = "tts_models/multilingual/multi-dataset/xtts_v1.1"
LANG_TTS = "zh-cn"
tts = TTS(model_name=model_path, progress_bar=True, gpu=True)
speaker_wav = "demos/speaker.wav"
output_wav_path = "demos/tts.wav"


class Chat:
    def __init__(self, init_prompt: Optional[str] = None,
                       llm: Optional[str] = "gpt-3.5-turbo"):
        
        self.init_prompt = init_prompt
        self.messages = []
        self.llm = llm

        if self.llm in ["gpt-3.5-turbo", "gpt-4"]:
            # openai
            self.api_key = OPENAI_KEY
            self.api_base = None
        elif self.llm in ["llama-2-70b-chat", "llama-2-13b-chat", "mistral-7b-instruct"]:
            # perplexity
            self.api_key = PPLX_KEY
            self.api_base = "https://api.perplexity.ai"
            
        self.client = OpenAI(api_key=self.api_key)
        
        if self.init_prompt is not None:
            self.messages.append({
                "role": "system",
                "content": init_prompt
            })
     
    def prompt(self, content: Optional[str] = None,
                     temperature: Optional[float] = 0.5) -> str:
        #print("content", content)
        if content is not None:
            self.messages.append({
                  "role": "user",
                  "content": content
            })
         
        response = self.client.chat.completions.create(
              model=self.llm,
              messages=self.messages,
              temperature=temperature
        )
       
        #response_content = response["choices"][0]["message"]["content"]
        response_content = response.choices[0].message.content
        
        self.messages.append({
              "role": "assistant",
              "content": response_content
        })
        print(self.messages)
         
        return response_content
    

chat_bots = {} 

def setup_prompt(chat_history, user_id, init_prompt, llm):
    if init_prompt == "":
        init_prompt = INIT_PROMPT
    
    chat_bots[user_id] = Chat(init_prompt=init_prompt, llm=llm)
    chat = chat_bots[user_id]
    bot_message = chat.prompt(content=warmup_reply, temperature=0.0)
    
    chat_history.append(("", bot_message))
    tts.tts_to_file(bot_message, speaker_wav=speaker_wav, language=LANG_TTS, file_path=output_wav_path)
    
    return chat_history, output_wav_path
    

def run_text_prompt(message, chat_history, user_id):
    chat = chat_bots[user_id]

    bot_message = chat.prompt(content=message)
    chat_history.append((message, bot_message))

    tts.tts_to_file(bot_message, speaker_wav=speaker_wav, language=LANG_TTS, file_path=output_wav_path)
    
    if len(chat_history) > 5:
        bot_message = chat.prompt(content=grade_reply, temperature=0.1)
        chat_history.append(("", bot_message))
    
    return "", chat_history, output_wav_path

stt = OpenAI(api_key=OPENAI_KEY)
def run_audio_prompt(audio, chat_history, user_id):
    if audio is None:
        return None, chat_history
    
    #os.rename(audio, audio + ".wav")
    #audio = audio + ".wav"

    with open(audio, "rb") as audio_file:
        transcript = stt.audio.transcriptions.create(model="whisper-1", file=audio_file, language=LANG)
   
    print(transcript) 
    message_transcription = transcript.text
    
    _, chat_history, output_wav_path = run_text_prompt(message_transcription, chat_history, user_id)
    
    return None, chat_history, output_wav_path


CSS ="""
.contain { display: flex; flex-direction: column; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; }
"""

with gr.Blocks(css=CSS) as demo:

    gr.Markdown("# Voice-ChatGPT")
    gr.Markdown("""
                    #### 1. 初始化(必要)：
                    * 1-1. 填入 User ID、LLM、Init. prompt，並按 setup。
                    #### 2. 開始聊天: 
                    * 2-1. 文字輸入，在 Text Message 方塊輸入文字，按下 Enter 送出。
                    * 2-2. 語音輸入，按下 Record from Microphone，並點 Send Audio 按鈕。
                    #### 3. 其他
                    * 3-1. 按下 clear 清除對話紀錄，並從步驟 1 開始。
                    * 3-2. [LLM] **openai**: gpt-3.5-turbo, gpt-4. **perplexity**: llama-2-70b-chat, llama-2-13b-chat, mistral-7b-instruct
                """
                )
                
    
    with gr.Row():
        with gr.Column():
            user_id = gr.Textbox(value=USER_ID, label="User ID, multi-turn coversation)")
            llm = gr.Dropdown(["gpt-3.5-turbo", "gpt-4"], value="gpt-3.5-turbo", label="LLM")
            init_prompt = gr.Textbox(value=INIT_PROMPT, label="Init. prompt")
            setup_button = gr.Button("Setup", interactive=True)
            msg = gr.Textbox(label="Text Message")
            
            with gr.Row():
                audio = gr.Audio(source="microphone", type="filepath")        
                send_audio_button = gr.Button("Send Audio", interactive=True)
            
        with gr.Column():
            chatbot = gr.Chatbot(elem_id="chatbot")
            syn_audio = gr.Audio(label="Synthesised Audio",autoplay=True)
            clear = gr.ClearButton([chatbot])
    
    setup_button.click(setup_prompt, [chatbot, user_id, init_prompt, llm], [chatbot, syn_audio])
    msg.submit(run_text_prompt, [msg, chatbot, user_id], [msg, chatbot, syn_audio])
    
    send_audio_button.click(run_audio_prompt, [audio, chatbot, user_id], [audio, chatbot, syn_audio])

demo.queue().launch(share=True)
