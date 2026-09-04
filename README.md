# Multitask NLP System 🤖

A Python-based Multitask Natural Language Processing (NLP) System that combines multiple NLP tasks into a single application.

The project provides both a command-line interface (CLI) and a Flask-based web interface for performing different NLP tasks using pre-trained Transformer models.

---

## 🚀 Features

The Multitask NLP System currently supports:

### 1. Next Word Prediction
Predicts the most probable next words for a given text prompt using GPT-2.

- Displays Top-K predictions
- Shows probability/confidence for each prediction
- Uses Transformer-based language modeling

### 2. Perplexity Evaluation
Evaluates the language modeling performance of GPT-2 using perplexity.

- Calculates test loss
- Calculates perplexity
- Supports custom text evaluation

### 3. Multilingual Text Translation
Provides translation between:

- 🇬🇧 English
- 🇮🇳 Tamil
- 🇮🇳 Telugu

Supported translation directions:

- English → Tamil
- English → Telugu
- Tamil → English
- Telugu → English
- Tamil → Telugu
- Telugu → Tamil

The system uses Meta's NLLB multilingual translation model.

### 4. Sentiment Analysis
Classifies English text as:

- POSITIVE
- NEGATIVE

The system also provides a confidence score.

### 5. Speech Translation 🎤
The web application supports speech-to-text followed by multilingual translation.

Speech can be recorded through the browser microphone.

Supported languages:

- English
- Tamil
- Telugu

The speech pipeline uses Whisper for speech recognition and NLLB for translation.

---

## 🧠 Models Used

| Task | Model |
|---|---|
| Next Word Prediction | `openai-community/gpt2` |
| Perplexity Evaluation | `openai-community/gpt2` |
| Text Translation | `facebook/nllb-200-distilled-600M` |
| Sentiment Analysis | `distilbert-base-uncased-finetuned-sst-2-english` |
| Speech Recognition | `openai/whisper-small` |

The models are downloaded automatically from Hugging Face when they are loaded for the first time.

---

## 🏗️ Project Architecture

```text
                    MULTITASK NLP SYSTEM
                            │
             ┌──────────────┴──────────────┐
             │                             │
        Command Line                   Web Interface
          Interface                      Flask
             │                             │
             └──────────────┬──────────────┘
                            │
                    NLP Processing
                            │
       ┌────────────────────┼────────────────────┐
       │                    │                    │
   GPT-2                NLLB-200            DistilBERT
       │                    │                    │
       │                    │                    │
 Next Word            Translation          Sentiment
 Prediction           EN/TA/TE             Analysis
       │
 Perplexity
 Evaluation

                            │
                     Speech Pipeline
                            │
                      Browser Microphone
                            │
                         Whisper
                      Speech-to-Text


🖥️ Software Requirements

Install the following software on the new system:

1. Python

Python 3.10 or newer is recommended.

Check Python installation:

python --version

or:

py --version
2. Git

Git is required if you want to clone this repository.

Check Git:

git --version
3. Web Browser

A modern browser such as:

Google Chrome
Microsoft Edge
Mozilla Firefox

A browser with microphone access is required for the speech translation feature.

📦 Installation
Step 1: Clone the Repository
git clone <YOUR_GITHUB_REPOSITORY_URL>

Move into the project directory:

cd Multitask-nlp
Step 2: Create a Virtual Environment

Windows:

python -m venv venv

Activate it:

venv\Scripts\activate

Linux/macOS:

python3 -m venv venv

Activate:

source venv/bin/activate
Step 3: Install Dependencies
pip install -r requirements.txt

The project currently requires:

Flask
PyTorch
Transformers
Matplotlib
SoundFile
ImageIO-FFmpeg
🤗 Transformer Models

The application uses pre-trained models from Hugging Face.

The following models are downloaded automatically when the corresponding Python modules are executed:

openai-community/gpt2
facebook/nllb-200-distilled-600M
distilbert-base-uncased-finetuned-sst-2-english
openai/whisper-small
Important

The first execution can take longer because the models need to be downloaded and cached locally.

A stable internet connection is recommended during the first run.

After the models have been downloaded, subsequent runs can load them from the local Hugging Face cache.

▶️ Running the Command-Line Application

Run:

python main.py

The CLI provides:

==================================================
             MULTITASK NLP SYSTEM
==================================================
1. Next Word Prediction
2. Perplexity Evaluation
3. Language Translation
4. Sentiment Analysis
5. Exit
==================================================

Enter the required option.

🌐 Running the Web Application

Start the Flask server:

python app.py

You should see:

Starting Multitask NLP System Web Server...
Open http://127.0.0.1:5000 in your browser.

Open the following address in your browser:

http://127.0.0.1:5000
                            │
                         NLLB-200
                        Translation
