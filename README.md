# Mayu Chatbot

Mayu is a Thai language chatbot built on LangGraph with vLLM for inference. Users can customize the chatbot by modifying `config.yaml` to use their own API settings.
Works with OpenAI API
## Features

### Dual Memory System
Mayu utilizes two separate memory types:

- **Temporary Memory**: Stores the 10 most recent conversation exchanges
- **Core Memory**: Permanent storage for critical user information including:
  - User's name
  - Hobbies and interests
  - Food preferences
  - General preferences

### Heart Level System
Mayu has an emotional engagement system with a heart level ranging from 0 (super hate) to 10 (maximum affection).

- **Starting Level**: 3
- **Progression**: Increases when users interact kindly with Mayu
- **Evaluation Method**: Hybrid approach combining:
  - LLM-based prompt analysis (digests user input into main ideas)
  - TextBlob NLP processing for sentiment analysis

### Configurable Settings
- `short_term_chat_history_num`: Number of recent memories to retain (default: 10)

## Setup and Usage

1. **Configuration**: Update `config.yaml` with your API credentials and preferred model
2. **Run**: Execute the following command:
```bash
   uv run Mayu.py
```

## Requirements
- LangGraph
- vLLM
- TextBlob