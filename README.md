# Voice-ChatGPT

This project integrates OpenAI's GPT and TTS (Text-to-Speech) functionality with a user-friendly interface provided by Gradio. It allows users to interact with a chatbot via text or audio input and receive synthesized audio responses.

## Features

1. **Chatbot Interaction**
   - Supports multi-turn conversations using OpenAI's GPT models.
   - Offers text and audio input options.
   - Provides grammar suggestions and text simplifications.

2. **Text-to-Speech (TTS)**
   - Choose from different TTS models and voices to synthesize responses.
   - Adjustable speaking rate.

3. **Chat History Saving**
   - Save chat history in JSON and Excel formats.
   - Includes the user's input, chatbot responses, grammar suggestions, and text simplifications.

## Setup Instructions

1. Install the required packages:
   ```bash
   conda create --name voice-chatgpt python=3.10
   pip install -r requirements.txt
   ```

2. Ensure `ffmpeg` is installed and available in your system's path.

3. Set up the necessary environment variables, such as:
   - `OPENAI_KEY` (for access to OpenAI's API)
   - `WARMUP_REPLY` (for the initial system response)
   - `MODEL_LIST` (for available LLMs)
   - `INIT_PROMPT` (for user instructions)
   - `TS_PROMPT` (for text simplification)
   - `RUBRIC_PROMPT` (for the end of dialogue)
   - `GRADE_PROMPT` (for grammar checking; not used in this version)

   These should be configured in `config_zhv5_1.py`.

## Usage (Web)

1. **Initialization**
   - Fill in the required fields (User ID, LLM, Prompts, TTS Models, etc.).
   - Click "Setup" to initialize the chatbot and TTS systems.

2. **Text Input**
   - Type your message in the "Text Message" box and press Enter to send it.
   
3. **Audio Input**
   - Record your message using the microphone and click "Send Audio" to submit it.

4. **Save Chat History**
   - Enter a filename and click "Save" to store the chat history.

## Project Files

- `config_zhv5_1.py`: Configuration file containing API keys, prompts, and other parameters.
- `appv6_saved.py`: Gradio-based UI for chatbot and TTS interactions.
- `chat_history`: Folder where chat history files are saved.

## Codes
- Please see in [https://github.com/teinhonglo/voice-chatgpt/blob/master/CODES.md](https://github.com/teinhonglo/voice-chatgpt/blob/master/CODES.md)

## External Links

- [OpenAI Models Documentation](https://platform.openai.com/docs/models)
