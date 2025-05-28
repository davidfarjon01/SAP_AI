import os
import fitz
import openai
import faiss
import pickle
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
model = SentenceTransformer('all-MiniLM-L6-v2')

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# def embed_text(text_list):
#     response = openai.embeddings.create(
#         model=EMBEDDING_MODEL,
#         input=text_list
#     )
#     return [d.embedding for d in response.data]

def embed_text(text_list):
    # Le modèle retourne un numpy array, on convertit en liste python
    embeddings = model.encode(text_list)
    return embeddings.tolist()

def load_or_create_index(pdf_folder, index_path):
    if os.path.exists(index_path) and os.path.exists(index_path + ".pkl"):
        print("[INFO] Loading existing index...")
        index = faiss.read_index(index_path)
        with open(index_path + ".pkl", "rb") as f:
            documents = pickle.load(f)
    else:
        print("[INFO] Creating new index...")
        texts = []
        documents = []
        for filename in os.listdir(pdf_folder):
            if filename.endswith(".pdf"):
                full_path = os.path.join(pdf_folder, filename)
                raw_text = extract_text_from_pdf(full_path)
                chunks = chunk_text(raw_text)
                texts.extend(chunks)
                documents.extend([(filename, chunk) for chunk in chunks])

        embeddings = embed_text(texts)
        dim = len(embeddings[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype("float32"))

        faiss.write_index(index, index_path)
        with open(index_path + ".pkl", "wb") as f:
            pickle.dump(documents, f)

    return index, documents

def query_index(prompt, index, documents, k=5):
    embedded_prompt = embed_text([prompt])[0]
    D, I = index.search(np.array([embedded_prompt]).astype("float32"), k)
    relevant_chunks = [documents[i][1] for i in I[0]]
    context = "\n---\n".join(relevant_chunks)

    system_message = "Tu es un assistant qui répond uniquement selon les documents fournis."
    full_prompt = f"{system_message}\n\nContexte:\n{context}\n\nQuestion: {prompt}"

    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": full_prompt}
        ]
    )
    return response.choices[0].message.content.strip()
