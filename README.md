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

## Command Line Usage

For command-line usage with images from the `raw/` folder:

1. Create a DeepSeek API key and save it as an environment variable called "DEEPSEEK_API_KEY"
2. Choose an image name from raw and change parameter on line 19 of pipeline.py to match the image name
3. Run `python3 pipeline.py`
4. Image will run through the denoising --> OCR --> LLM
5. .tex and .pdf files of the corresponding LaTeX notes will be saved to notes_out folder

## Project Structure

- **`app.py`** - Flask web application
- **`pipeline.py`** - Command-line pipeline script
- **`denoise_pipeline.py`** - Image denoising and enhancement functions
- **`templates/index.html`** - Web app frontend
- **`raw/`** - Raw images, unprocessed
- **`processed/`** - Processed images (temporary)
- **`notes_out/`** - Generated LaTeX and PDF files
- **`old/`** - Old files that are NOT currently in use
