import os
import secrets
from typing import Optional
from collections import defaultdict
import json

import gradio as gr
import openai
from openai import OpenAI
import io
from pydub import AudioSegment
from pydub.utils import which

import hashlib
import pandas as pd
import time

# All paramters are defined in this file
from config_zhv5_1 import *

user_id_length = 13
openai.api_key = OPENAI_KEY

output_wav_path = "demos/ttsv4_saved.mp3"
warmup_reply = WARMUP_REPLY
model_list = MODEL_LIST
AudioSegment.converter = which("ffmpeg")

'''
1-1. Chatbot
'''
class Chat:
    def __init__(self, init_prompt: Optional[str] = None,
                       llm: Optional[str] = "gpt-3.5-turbo",
                       rubric_reply: Optional[str] = ""):
        
        self.init_prompt = init_prompt
        self.messages = []
        self.llm = llm
        self.rubric_reply = rubric_reply

        if self.llm in model_list:
            # openai
            self.api_key = OPENAI_KEY
            self.api_base = None
            
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
        response_content = response.choices[0].message.content
        
        self.messages.append({
              "role": "assistant",
              "content": response_content
        })
        
        print(self.messages)

        return response_content

'''
1-2. Text Simplification
'''
class ChatTS:
    def __init__(self, init_prompt: Optional[str] = None,
                       llm: Optional[str] = "gpt-3.5-turbo"):
        
        self.init_prompt = init_prompt
        self.messages = []
        self.llm = llm

        if self.llm in model_list:
            # openai
            self.api_key = OPENAI_KEY
            self.api_base = None
            
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
        
        # remove the last response
        self.messages = self.messages[:-1]

        return response_content
    
'''
1-3. TTS
'''
class TTS:
    def __init__(self, voice: Optional[str] = "alloy",
                       model: Optional[str] = "tts-1-hd",
                       speed: Optional[str] = "1.0"):
        
        self.voice = voice
        self.model = model
        self.api_key = OPENAI_KEY
        self.speed = float(speed)
        self.client = OpenAI(api_key=self.api_key)
     
    def synthesis(self, text: Optional[str] = "",
                        speed: Optional[str] = "1.0") -> str:
        response = self.client.audio.speech.create(
            model=self.model, # tts-1-hd or tts-1
            voice=self.voice,
            speed=speed,
            input=text,
        )

        # Convert the binary response content to a byte stream
        byte_stream = io.BytesIO(response.content)
        response.stream_to_file(output_wav_path)
        
        # Read the audio data from the byte stream
        #audio = AudioSegment.from_file(byte_stream, format="mp3")
        #audio.export(output_wav_path, format="mp3")
         
        return output_wav_path


chat_bots = defaultdict(dict)
tts_bots = {}

'''
1. Init.
'''
def setup_prompt(chat_history, user_id, init_prompt, grade_prompt, ts_prompt, rubric_prompt, llm, tts_mdl, tts_spk, tts_speed):

    chat_history = []    

    if init_prompt == "":
        init_prompt = INIT_PROMPT
    
    if grade_prompt == "":
        grade_prompt = GRADE_PROMPT
    
    if ts_prompt == "":
        ts_prompt = TS_PROMPT

    if rubric_prompt != "":
        rubric_reply = rubric_prompt
    
    chat_bots[user_id]["chat"] = Chat(init_prompt=init_prompt, llm=llm, rubric_reply=rubric_reply)
    chat_bots[user_id]["ts"] = ChatTS(init_prompt=ts_prompt, llm=llm)
    tts_bots[user_id] = TTS(voice=tts_spk, model=tts_mdl)
    
    chat = chat_bots[user_id]["chat"]
    tts = tts_bots[user_id]
    bot_message = chat.prompt(content=warmup_reply, temperature=0.0)
    
    chat_history.append(("", bot_message))
    
    output_wav_path = tts.synthesis(bot_message, tts_speed)
    
    return chat_history, output_wav_path
    
'''
2-1. Text as the input of Chatbot
'''
def run_text_prompt(message, chat_history, user_id, tts_speed):
    chat = chat_bots[user_id]["chat"]
    chat_ts = chat_bots[user_id]["ts"]
    tts = tts_bots[user_id]

    if len(chat_history) <= 10:
        bot_message = chat.prompt(content=message)
        bot_message_grade = "EMPTY"
        bot_message_ts = chat_ts.prompt(content=bot_message)
        
        chat_history.append((message, f"{bot_message} </br></br> {bot_message_grade} </br></br> {bot_message_ts}"))

        output_wav_path = tts.synthesis(bot_message, tts_speed)
    
        if len(chat_history) == 11:
            bot_message = chat.prompt(content=chat.rubric_reply, temperature=0.1)
            chat_history.append(("", bot_message))
    else:
        output_wav_path = None
    
    return "", chat_history, output_wav_path

'''
2-2. Audio as the input of Chatbot
'''
stt = OpenAI(api_key=OPENAI_KEY)
def run_audio_prompt(audio, chat_history, user_id, tts_speed):
    if audio is None:
        return None, chat_history, None

    with open(audio, "rb") as audio_file:      
        transcript = stt.audio.transcriptions.create(model="whisper-1", file=audio_file, language=LANG)
   
    message_transcription = transcript.text
    
    _, chat_history, output_wav_path = run_text_prompt(message_transcription, chat_history, user_id, tts_speed)
    
    return None, chat_history, output_wav_path

'''
Saved Chatbot history
'''
CHAT_TEMP_DIR = "chat_history"
def save_chat_history(user_id, chat_history, saved_fname, init_prompt, grade_prompt, ts_prompt, rubric_prompt):
    
    chat = chat_bots[user_id]["chat"]
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    chat_temp_json_fn = os.path.join(CHAT_TEMP_DIR, saved_fname + "_" + timestr + ".json")
    chat_temp_csv_fn = os.path.join(CHAT_TEMP_DIR, saved_fname + "_" + timestr + ".xlsx")
    chat_temp_dir = os.path.dirname(chat_temp_json_fn)
    
    if not os.path.exists(chat_temp_dir):
        os.makedirs(chat_temp_dir)

    chat_history_json = {"chat_history": []}
    chat_history_csv = {"chatbot": [], "user": [], "grammar_suggestion": [], "text_simplification": []}
    prompt_csv = {"init_prompt": [init_prompt], "grade_prompt": [grade_prompt], "ts_prompt": [ts_prompt], "rubric_prompt": [rubric_prompt]}
    
    for i, (user_response, chatbot_response) in enumerate(chat_history):
        
        chat_json = {   "user": user_response,
                        "chatbot": {
                            "response": "",
                            "grammar_suggestion": "",
                            "text_simplification": "",
                        }
                    }    
        # 第一次的回覆沒有 grammar_suggestion 和 text_simplification
        if i > 0:
            # 使用 "</br></br>" 分割字符串
            splitted = chatbot_response.split("</br></br>")
            try:
                response = splitted[0]
                grammar_suggestion = splitted[1]
                ts_cleaned = splitted[2].replace('\\n', '').replace('\\"', '"')
                text_simplification = json.loads(ts_cleaned)
            except:
                response = chatbot_response
                grammar_suggestion = ""
                text_simplification = ""
        else:
            response = chatbot_response
            grammar_suggestion = ""
            text_simplification = ""
        
        chat_json["chatbot"]["response"] = response
        chat_json["chatbot"]["grammar_suggestion"] = grammar_suggestion
        chat_json["chatbot"]["text_simplification"] = text_simplification
            
        chat_history_json["chat_history"].append(chat_json)
        chat_history_csv["chatbot"].append(chat_json["chatbot"]["response"])
        chat_history_csv["user"].append(user_response)
        chat_history_csv["grammar_suggestion"].append(chat_json["chatbot"]["grammar_suggestion"])
        chat_history_csv["text_simplification"].append(chat_json["chatbot"]["text_simplification"])
    
    # Write chat messages to the file
    try:
        with open(chat_temp_json_fn, "w", encoding='utf-8') as fn:
            json.dump(chat_history_json, fn, ensure_ascii=False, indent=4)
        
        with pd.ExcelWriter(chat_temp_csv_fn, engine='openpyxl') as writer:
            chat_history_csv_df = pd.DataFrame.from_dict(chat_history_csv)
            chat_history_csv_df.to_excel(writer, sheet_name="chat_history", index=False)

            prompt_csv_df = pd.DataFrame.from_dict(prompt_csv)
            prompt_csv_df.to_excel(writer, sheet_name="prompt", index=False)
        
        print(f"Chat history saved to {chat_temp_dir}")
    except Exception as e:
        print(f"Error saving chat history: {e}")



'''
0. Display

[Audio] 
[Text]

'''
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
                    * 3-2. [LLM] **openai**: [https://platform.openai.com/docs/models](https://platform.openai.com/docs/models)
                    * 3-3. Download Link: [https://140.122.184.167:5567/sharing/WWDgQuqw4](https://140.122.184.167:5567/sharing/WWDgQuqw4)
                """
                )
                
    with gr.Row():
        with gr.Column():
            user_id = gr.Textbox(value=USER_ID, label="User ID, multi-turn coversation)")
            llm = gr.Dropdown(model_list, value="gpt-4o", label="LLM")
            tts_mdl = gr.Dropdown(["tts-1", "tts-1-hd"], value="tts-1", label="TTS Models")
            tts_spk = gr.Dropdown(["alloy", "echo", "fable", "onyx", "nova", "shimmer"], value="alloy", label="TTS Speakers")
            tts_speed = gr.Textbox(value=1, label="Speaking Rate")
            
            with gr.Row():
                init_prompt = gr.Textbox(value=INIT_PROMPT, label="prompt (init.)")
                grade_prompt = gr.Textbox(value=GRADE_PROMPT, label="prompt (grammar)")
                ts_prompt = gr.Textbox(value=TS_PROMPT, label="prompt (text simplification)")
            
            rubric_prompt = gr.Textbox(value=RUBRIC_PROMPT, label="prompt (rubric)")
            
            setup_button = gr.Button("Setup", interactive=True)
            msg = gr.Textbox(label="Text Message")
            
            with gr.Row():
                audio = gr.Audio(source="microphone", type="filepath")        
                send_audio_button = gr.Button("Send Audio", interactive=True)
            
        with gr.Column():
            chatbot = gr.Chatbot(elem_id="chatbot")
            syn_audio = gr.Audio(label="Synthesised Audio", autoplay=True)
            
            with gr.Row():
                saved_fname = gr.Textbox(value=SAVED_FILENAME, label="filename")
                save_button = gr.Button("Saved", interactive=True)
            clear = gr.ClearButton([chatbot])
    
    setup_button.click(setup_prompt, [chatbot, user_id, init_prompt, grade_prompt, ts_prompt, rubric_prompt, llm, tts_mdl, tts_spk, tts_speed], [chatbot, syn_audio])
    msg.submit(run_text_prompt, [msg, chatbot, user_id, tts_speed], [msg, chatbot, syn_audio])
    
    send_audio_button.click(run_audio_prompt, [audio, chatbot, user_id, tts_speed], [audio, chatbot, syn_audio])
    save_button.click(save_chat_history, [user_id, chatbot, saved_fname, init_prompt, grade_prompt, ts_prompt, rubric_prompt])

demo.queue().launch(share=True)
