from dotenv import load_dotenv
from flask import Flask, render_template, request

from utils import load_or_create_index
from utils import query_index

load_dotenv()

app = Flask(__name__)

# Load FAISS index and documents on startup
INDEX_PATH = "vector_index.index"
DOCS_PATH = "pdfs"
index, documents = load_or_create_index(DOCS_PATH, INDEX_PATH)

@app.route("/", methods=["GET", "POST"])
def index_page():
    answer = ""
    if request.method == "POST":
        user_prompt = request.form.get("prompt")
        if user_prompt:
            answer = query_index(user_prompt, index, documents)
    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
