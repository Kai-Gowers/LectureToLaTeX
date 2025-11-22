# LectureToLaTeX

Convert blackboard notes into beautiful LaTeX documents using AI-powered OCR and transcription.

## Web App (Recommended)

The easiest way to use LectureToLaTeX is through the web interface:

1. **Set up your environment:**

   - Create a DeepSeek API key and save it as an environment variable:
     ```bash
     export DEEPSEEK_API_KEY="sk-your-key-here"
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
   - Upload an image of your blackboard notes
   - Wait for processing (denoising → OCR → LaTeX generation)
   - Download the generated `.tex` file (and `.pdf` if LaTeX is installed)

## Project Structure

- **`app.py`** - Flask web application
- **`denoise_pipeline.py`** - Image denoising and enhancement functions
- **`templates/index.html`** - Web app frontend
- **`notes_out/`** - Generated LaTeX and PDF files
- **`math_chatbot.py/`** - Chatbot

## Old Files

- **`old/`** - Old files that are NOT currently in use
