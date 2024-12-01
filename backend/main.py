import shutil
from typing import List
from fastapi import FastAPI, HTTPException, File, UploadFile, Body
from uuid import uuid4
from pathlib import Path
import os
from app.config_ai import OpenAIConfig
from app.cv_analysis import CVAnalysis
from app.schemas import ErrorResponse, CVAnalysisResponse, CVAnalysisInput

app = FastAPI()

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

API_KEY = "25417b9e73574c49965cad8f28ab4dd6"
API_VERSION = "2024-02-01"
AZURE_ENDPOINT = "https://openaitcuc.openai.azure.com/"
DEPLOYMENT_NAME = "corpu2-text-embedding-3-large"

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Hanya izinkan frontend tertentu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze-cv/", response_model=CVAnalysisResponse, responses={500: {"model": ErrorResponse}})
async def analyze_cv(
    file: UploadFile = File(...),
    job_description: str = Body(...),
    tags: List[str] = Body(...)
):
    try:
        # Save the uploaded file
        pdf_filename = f"{uuid4().hex}_{file.filename}"
        pdf_path = UPLOAD_DIR / pdf_filename

        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Perform CV analysis
        openai_config = OpenAIConfig(
            api_key=API_KEY,
            api_version=API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            deployment_name=DEPLOYMENT_NAME,
        )

        split_tags = []
        for tag in tags:
            split_tags.extend(tag.split(","))

        split_tags = [tag.strip() for tag in split_tags] 

        cv_analysis = CVAnalysis(
            pdf_file=str(pdf_path),
            description_job=job_description,
            tag_match=split_tags,
            openai_config=openai_config
        )

        tag_matched = cv_analysis.find_matching_tags()
        tag_found = cv_analysis.find_matching_tags(use_tag_match=False)
        similarity_score = cv_analysis.calculate_similarity_score()
        tag_matched_score = cv_analysis.calculate_tag_match_score()
        total_sscore   = cv_analysis.calculate_total_score()
        summary = cv_analysis.get_summary_from_cv()

        # Clean up the temporary file
        os.remove(pdf_path)

        return CVAnalysisResponse(
            summary=summary,
            tag_found=tag_found,
            tag_matched=tag_matched,
            similarity_score=similarity_score,
            tag_matched_score=tag_matched_score,
            total_score=total_sscore   
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CV: {str(e)}")
