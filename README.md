# LectureToLaTeX

How to run this on your machine:

1. Create a deepseek API key for yourself, save it as an environment variable called "DEEPSEEK_API_KEY"
2. Choose an image from pre_out and change parameter on line 18 of pipeline.py to match the image name
3. Run python3 pipeline.py
4. Image will run through the denoising --> OCR --> LLM
5. .tex and .pdf files of the corresponding LaTeX notes will be saved to notes_out folder

The "old" folder consists of old files that are NOT currently in use.
The "raw" folder consists of raw images, unprocessed.
The "processed" folder consists of the processed images.
