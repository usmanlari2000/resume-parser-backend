import os
import io
import json
import base64
import openai
import pdfplumber
import fitz
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

openai.api_key = OPENAI_API_KEY

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 5 * 1024 * 1024
MAX_FILES = 50
MAX_CHARACTERS = 5000

def parse_resume(text):
    prompt = (
        "You are a resume parser. Extract the following features in JSON format:\n"
        "If a value can't be found or inferred from the resume, use \"\" for strings, -1 for integers and floats, and [] for lists.\n"
        "- full_name (string, e.g., \"John Doe\")\n"
        "- email_address (string, e.g., \"john.doe@example.com\")\n"
        "- phone_number (string, e.g., \"+1 123 456 7890\")\n"
        "- location (list of objects with keys \"country\" and \"city\", e.g., [{ \"country\": \"United States\", \"city\": \"New York\" }, { \"country\": \"Germany\", \"city\": \"Berlin\" }])\n"
        "- linkedin_profile_url (string, e.g., \"https://www.linkedin.com/in/johndoe/\")\n"
        "- skills (list of strings, e.g., [\"Python\", \"TensorFlow\", \"Communication\"])\n"
        "- previous_job_titles (list of strings, e.g., [\"Software Engineer\", \"Data Scientist\"])\n"
        "- previous_companies (list of strings, e.g., [\"Google\", \"Microsoft\"])\n"
        "- total_years_of_professional_experience (integer, e.g., 3)\n"
        "- number_of_projects (integer, e.g., 2)\n"
        "- bachelors_degree_program (string, e.g., \"Computer Science\")\n"
        "- bachelors_gpa (float, e.g., 3.8)\n"
        "- masters_degree_program (string, e.g., \"Data Science\")\n"
        "- masters_gpa (float, e.g., 3.4)\n"
        "- languages (list of objects with keys \"language\" and \"proficiency_level\", where \"proficiency_level\" must be one of the following values: \"Elementary\", \"Working\", \"Fluent\", \"Native\", \"\" e.g., [{\"language\": \"French\", \"proficiency_level\": \"Native\"}, {\"language\": \"English\", \"proficiency_level\": \"Working\"}])\n"
        f"Resume text:\n{text[:MAX_CHARACTERS]}"
    )

    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You return structured JSON from resumes. Under no circumstances should you return anything other than valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as e:
        return {"error": f"Unable to decode OpenAI response as JSON: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}
    
def extract_image(file_bytes):
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                images = page.get_images(full=True)
                
                if images:
                    xref = images[0][0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    return base64.b64encode(image_bytes).decode("utf-8")
        return None
    except Exception as e:
        return None

@app.post("/upload")
async def upload_resumes(resumes: list[UploadFile] = File(...)):
    if len(resumes) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Max {MAX_FILES} files allowed per upload.")

    results = []
    
    for file in resumes:
        file_bytes = await file.read()

        if not file_bytes:
            continue

        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"Max allowed file size is {MAX_FILE_SIZE / (1024 * 1024)}MB.")

        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

        extracted = parse_resume(text)

        image = extract_image(file_bytes)

        extracted.update({
            "filename": file.filename,
            "image": image
        })
        results.append(extracted)

    return {"results": results}