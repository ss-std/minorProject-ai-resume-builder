from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from engine.llm_handler import AIHandler

app = FastAPI()

# ---  CORS SECTION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # React's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------------------

ai = AIHandler(provider="gemini")

class ResumeRequest(BaseModel):
    raw_text: str

@app.get("/")
def home():
    return {"message": "AI Resume Builder API is running!"}

@app.post("/generate-summary")
async def generate_summary(data: ResumeRequest):
    prompt = f"Create a professional 2-sentence resume summary for this: {data.raw_text}"
    result = ai.generate_content(prompt)
    return {"summary": result}