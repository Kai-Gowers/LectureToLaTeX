# LectureToLaTeX

Convert blackboard notes into beautiful LaTeX documents using AI-powered vision and transcription.

## Features

- **AI-Powered Transcription**: Uses GPT-4o Vision API to transcribe handwritten mathematical notes into LaTeX
- **Image Enhancement**: Automatic denoising and enhancement pipeline for better recognition
- **Multiple Image Support**: Upload multiple images to create a single combined LaTeX document
- **Math Chatbot**: Interactive chatbot with LaTeX rendering support for math help and explanations
- **PDF Preview**: Preview generated LaTeX documents as PDFs directly in the browser
- **History Management**: View, download, and manage your previously generated notes
- **Feedback System**: Submit feedback on generated LaTeX to improve quality

## Web App

The easiest way to use LectureToLaTeX is through the web interface:

1. **Set up your environment:**

   - Create an OpenAI API key and save it as an environment variable:
     ```bash
     export OPENAI_API_KEY="sk-your-key-here"
     ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web app:**

   ```bash
   python app.py
   ```

4. **Open your browser:**
   - Navigate to `http://localhost:5000`
   - Upload one or more images of your blackboard notes
   - Wait for processing (denoising → AI vision → LaTeX generation)
   - Preview, download, or chat about the generated `.tex` file (and `.pdf` if LaTeX is installed)

## Requirements

- Python 3.8+
- OpenAI API key (or DeepSeek API key as alternative)
- Tesseract OCR (for image processing)
- LaTeX distribution (optional, for PDF generation)

## Project Structure

- **`app.py`** - Flask web application (main server and API endpoints)
- **`denoise_pipeline.py`** - Image denoising and enhancement functions
- **`math_chatbot.py`** - Math chatbot with LaTeX rendering support 
- **`templates/index.html`** - Web app frontend (HTML, CSS, JavaScript with KaTeX for math rendering)
- **`requirements.txt`** - Python dependencies -
- **`notes_out/`** - Generated LaTeX (.tex) and PDF files
- **`notes_feedback/`** - Feedback files (JSONL format) for chatbot interactions
- **`static/`** - Static assets (images, logos, etc.)

## Contributions

- **`app.py`** - Alexandra, Kai, Tianli
- **`denoise_pipeline.py`** - Alexandra and Kai
- **`math_chatbot.py`** - Alexandra
- **`templates/index.html`** - Kai


