from flask import Flask, request, jsonify
import json

app = Flask(__name__)

# 시작 시 한 번만 로드
with open("rag_output.json", "r", encoding="utf-8") as f:
    qa_list = json.load(f)  # [{"question": "...", "answer":"..."}, ...]

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    q = data.get("question", "").strip().lower()
    # 단순 매칭 예시: 똑같은 question이 있을 때
    for item in qa_list:
        if item.get("question", "").strip().lower() == q:
            return jsonify({"answer": item.get("answer")})
    # 없으면 첫 번째 답변(혹은 “없음”)
    return jsonify({"answer": "죄송합니다, 해당 질문에 대한 답변이 없습니다."})

if __name__ == "__main__":
    app.run(port=8000, debug=True)
