import os
import json
import openai
import pdfplumber
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def parse_resume(text):
    prompt = (
        'You are a resume parser. Extract the following features in JSON format:\n'
        'If a value can\'t be found or inferred from the resume, use "" for strings, -1 for integers and floats, and [] for lists.\n'
        '- full_name (string, e.g., "John Doe")\n'
        '- email_address (string, e.g., "john.doe@example.com")\n'
        '- phone_number (string, e.g., "+1 123 456 7890")\n'
        '- linkedin_profile_url (string, e.g., "https://www.linkedin.com/in/johndoe/")\n'
        '- skills (list of strings, e.g., ["Python", "TensorFlow", "Communication"])\n'
        '- previous_job_titles (list of strings, e.g., ["Software Engineer", "Data Scientist"])\n'
        '- previous_companies (list of strings, e.g., ["Google", "Microsoft"])\n'
        '- total_years_of_professional_experience (integer, e.g., 3)\n'
        '- number_of_projects (integer, e.g., 2)\n'
        '- bachelors_degree_program (string, e.g., "Computer Science")\n'
        '- bachelors_gpa (float, e.g., 3.8)\n'
        '- masters_degree_program (string, e.g., "Data Science")\n'
        '- masters_gpa (float, e.g., 3.4)\n'
        '- languages (list of objects with keys "language" and "proficiency_level", where "proficiency_level" can only be one of the following values: "Elementary", "Working", "Fluent", "Native", or "" e.g., [{"language": "French", "proficiency_level": "Native"}, {"language": "English", "proficiency_level": "Working"}])\n'
        f'Resume text:\n{text[:3500]}'
    )

    try:
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[
                {'role': 'system', 'content': 'You return structured JSON from resumes. Under no circumstances should you return anything other than valid JSON.'},
                {'role': 'user', 'content': prompt}
            ],
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as err:
        return {'error': str(err)}
    
@app.post("/upload")
async def upload_resumes(resumes: list[UploadFile] = File(...)):
    results = []
    for file in resumes:
        with pdfplumber.open(file.file) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text() is not None])
            extracted = parse_resume(text)
            extracted.update({"filename": file.filename})
            results.append(extracted)
    return {"results": results}