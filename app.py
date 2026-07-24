"""
app.py
------
Flask web server exposing:
  GET  /              -> chat UI
  POST /api/chat      -> {"message": "..."} -> {"answer": "...", "matched_question": "...", "score": ...}
"""

from flask import Flask, jsonify, render_template, request

from faq_matcher import FAQMatcher

app = Flask(__name__)
matcher = FAQMatcher(faqs_path="faqs.json", confidence_threshold=0.15)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"answer": "Please type a question.", "matched_question": None, "score": 0.0})

    result = matcher.best_match(user_message)
    if result is None:
        return jsonify(
            {
                "answer": (
                    "I'm sorry, I couldn't find a confident answer to that. "
                    "Could you rephrase, or contact support@example.com?"
                ),
                "matched_question": None,
                "score": 0.0,
            }
        )

    return jsonify(
        {
            "answer": result.answer,
            "matched_question": result.question,
            "score": round(result.score, 3),
        }
    )


@app.route("/api/faqs")
def list_faqs():
    """Handy endpoint to show all available FAQs (e.g. for a sidebar)."""
    return jsonify(matcher.faqs)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
