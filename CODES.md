## Code Explanation

### Classes

#### 1. `Chat`
- **Purpose**: This class initializes and handles interactions with the GPT-based chatbot, allowing the user to send prompts and receive responses.
- **Attributes**:
  - `init_prompt`: A prompt used to set the initial context for the conversation (optional).
  - `llm`: The language model to use (default is `"gpt-3.5-turbo"`).
  - `rubric_reply`: Additional instructions or rubrics for the chatbot's replies (optional).
  - `messages`: A list of all the messages exchanged with the chatbot during the session.
  - `client`: An instance of OpenAI's API client.
  
- **Methods**:
  - `prompt(content, temperature)`: Sends a user prompt to the GPT model, appends the response to the message history, and returns the generated response.
  
#### 2. `ChatTS`
- **Purpose**: Similar to the `Chat` class, this class is specifically used for handling text simplification tasks.
- **Attributes**:
  - Same as the `Chat` class (with `llm` defaulting to `"gpt-3.5-turbo"`).

- **Methods**:
  - `prompt(content, temperature)`: Sends a prompt to the model to simplify the text and returns the simplified version. The last response is removed to ensure the simplification occurs on a clean slate.

#### 3. `TTS`
- **Purpose**: This class handles the text-to-speech (TTS) functionality, converting chatbot responses into audio.
- **Attributes**:
  - `voice`: The voice model used for speech synthesis (default is `"alloy"`).
  - `model`: The TTS model to use (default is `"tts-1-hd"`).
  - `speed`: Adjusts the speaking speed (default is `"1.0"`).
  - `client`: An instance of OpenAI's API client.

- **Methods**:
  - `synthesis(text, speed)`: Converts the provided text into speech and saves the output as an audio file.

### Functions

#### 1. `setup_prompt(chat_history, user_id, init_prompt, grade_prompt, ts_prompt, rubric_prompt, llm, tts_mdl, tts_spk, tts_speed)`
- **Purpose**: Initializes the chatbot and TTS instances for a specific user. It also sends a warm-up message to the chatbot.
- **Parameters**:
  - `chat_history`: Stores the dialogue history.
  - `user_id`: Unique ID for each user.
  - `init_prompt`, `grade_prompt`, `ts_prompt`, `rubric_prompt`: Various prompts used to set up the conversation context.
  - `llm`, `tts_mdl`, `tts_spk`, `tts_speed`: Model and speed configuration for TTS.
  
- **Returns**: Updated chat history and the path to the synthesized audio file.

#### 2. `run_text_prompt(message, chat_history, user_id, tts_speed)`
- **Purpose**: Handles text-based input from the user, generating responses from both the GPT chatbot and text simplification system. It also triggers TTS synthesis.
- **Parameters**:
  - `message`: The user's input message.
  - `chat_history`: Conversation history.
  - `user_id`: Unique user identifier.
  - `tts_speed`: Speed of the synthesized speech.

- **Returns**: Empty string (reset message input box), updated chat history, and the path to the synthesized audio.

#### 3. `run_audio_prompt(audio, chat_history, user_id, tts_speed)`
- **Purpose**: Handles audio input from the user by transcribing the speech into text using OpenAI’s Whisper model and then passing the transcribed text to the chatbot.
- **Parameters**:
  - `audio`: Audio input file.
  - `chat_history`: Conversation history.
  - `user_id`: Unique user identifier.
  - `tts_speed`: Speed of the synthesized speech.

- **Returns**: Updated chat history and the path to the synthesized audio.

#### 4. `save_chat_history(user_id, chat_history, saved_fname, init_prompt, grade_prompt, ts_prompt, rubric_prompt)`
- **Purpose**: Saves the conversation history into JSON and Excel formats. It logs the user inputs, chatbot responses, grammar suggestions, and text simplifications.
- **Parameters**:
  - `user_id`: Unique identifier for the user.
  - `chat_history`: The conversation history to be saved.
  - `saved_fname`: Filename to save the history under.
  - `init_prompt`, `grade_prompt`, `ts_prompt`, `rubric_prompt`: Prompts used for initializing and guiding the conversation.

- **Returns**: None (saves files directly to disk).
