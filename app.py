from flask import Flask, request, jsonify
from rag_chain import invoke_and_save
import os
import uuid

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get("question")
    session_id=str(uuid.uuid4())
    if not question:
        return jsonify({"error": "Question is required"}), 400

    try:
        answer = invoke_and_save(session_id,question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    #os.environ["OPENAI_API_KEY"] = "your-openai-key-here"  # Or set it in .env
    app.run(debug=True)