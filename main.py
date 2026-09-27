from fastapi import FastAPI, HTTPException
import spacy

app = FastAPI()
nlp = spacy.load("en_core_web_lg")

@app.get("/")
def root():
    return {"message": "Hello, FastAPI with UV!"}

@app.get("/embedding")
def embedding(word: str):
    doc = nlp(word)
    if not doc.has_vector:
        raise HTTPException(status_code=404, detail="No vector for this word")
    return {
        "word": word,
        "dimension": len(doc.vector),
        "embedding": doc.vector.tolist(),
    }