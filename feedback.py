# feedback.py
import os
import json
from datetime import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

# Directory to store RLHF-style feedback
FEEDBACK_DIR = "notes_feedback"
os.makedirs(FEEDBACK_DIR, exist_ok=True)


@app.route('/')
def index():
    return jsonify({
        "message": "RLHF Feedback API is running.",
        "usage": "POST /feedback with JSON payload."
    })


@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """
    Collect user feedback on generated LaTeX.

    Expected JSON:
    {
        "note_name": "notes_2025-01-01_12-34-56",
        "rating_transcription": 1-5,
        "rating_explanation": 1-5,
        "corrected_latex": "...optional corrected LaTeX...",
        "comments": "optional free-text comments"
    }
    """
    try:
        data = request.json or {}

        note_name = data.get("note_name", "").strip()
        if not note_name:
            return jsonify({"error": "note_name is required"}), 400

        # Sanitize ratings
        def sanitize_rating(x):
            if x is None:
                return None
            try:
                x = int(x)
                return max(1, min(5, x))
            except:
                return None

        record = {
            "note_name": note_name,
            "rating_transcription": sanitize_rating(data.get("rating_transcription")),
            "rating_explanation": sanitize_rating(data.get("rating_explanation")),
            "corrected_latex": data.get("corrected_latex", ""),
            "comments": data.get("comments", ""),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        # Store as JSONL (one line per feedback)
        filename = os.path.join(FEEDBACK_DIR, f"{note_name}_feedback.jsonl")
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[INFO] Saved feedback for {note_name}")
        return jsonify({"success": True, "saved_to": filename})

    except Exception as e:
        print(f"[ERROR] Feedback save failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=6000)
