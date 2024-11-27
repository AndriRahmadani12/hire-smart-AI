from app.config_ai import OpenAIConfig
from typing import List
from app.tag import tag
from app.text_processor import TextProcessor
from app.pdf_processor import PDFProcessor
import re
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CVAnalysis:
    def __init__(self, pdf_file: str, description_job: str, tag_match: List[str], openai_config: OpenAIConfig):
        self.pdf_file = pdf_file
        self.description_job = description_job
        self.tag_match = tag_match
        print(f"Tag match: {self.tag_match}")
        self.openai_config = openai_config
        self.tag_list = tag()
        
        self.text_processor = TextProcessor()
        self.cv_text = PDFProcessor.extract_text(pdf_file)
        
    def find_matching_tags(self, use_tag_match: bool = True) -> List[str]:
        text = self.cv_text.lower()
        keywords = self.tag_match if use_tag_match else self.tag_list
        return [keyword.lower() for keyword in keywords 
                if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text)]

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        return [text[i:i + chunk_size] 
                for i in range(0, len(text), chunk_size - overlap)]

    def get_embedding(self, text: str) -> List[float]:
        try:
            client = openai.AzureOpenAI(
                api_key=self.openai_config.api_key,
                api_version=self.openai_config.api_version,
                azure_endpoint=self.openai_config.azure_endpoint
            )
            response = client.embeddings.create(
                input=text, 
                model=self.openai_config.deployment_name
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Embedding generation failed: {str(e)}")

    def calculate_similarity_score(self) -> float:
        try:
            pdf_embedding = self._get_pdf_mean_embedding()
            job_embedding = self._get_job_description_embedding()
            
            pdf_vector = np.array(pdf_embedding).reshape(1, -1)
            job_vector = np.array(job_embedding).reshape(1, -1)
            
            similarity = cosine_similarity(pdf_vector, job_vector)[0][0]
            return round(similarity * 100, 2)
        except Exception as e:
            raise Exception(f"Similarity calculation failed: {str(e)}")
        
    def calculate_tag_match_score(self) -> float:
        matching_tags = self.find_matching_tags()
        return len(matching_tags) / len(self.tag_match) * 100
    
    def calculate_total_score(self) -> float:
        similarity_score = self.calculate_similarity_score()
        tag_match_score = self.calculate_tag_match_score()
        return (similarity_score + tag_match_score) / 2

    def _get_pdf_mean_embedding(self) -> np.ndarray:
        chunks = self.chunk_text(self.cv_text)
        embeddings = [self.get_embedding(self.text_processor.clean_text(chunk)) 
                     for chunk in chunks]
        return np.mean(embeddings, axis=0)

    def _get_job_description_embedding(self) -> List[float]:
        cleaned_description = self.text_processor.clean_text(self.description_job)
        return self.get_embedding(cleaned_description)
    
    def get_summary_from_cv(self):
        try:
            
            prompt = f"""Please provide a concise summary of the following CV:
                {self.cv_text}

                write in paragraphed format. The summary should be no more than 500 words.
            """
            client = openai.AzureOpenAI(
                api_key=self.openai_config.api_key,
                api_version=self.openai_config.api_version,
                azure_endpoint=self.openai_config.azure_endpoint
            )

            response = client.chat.completions.create(
                model="corpu-text-gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful HR assistant that summarizes CVs."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.replace(" .", ".").strip()
        except Exception as e:
            raise Exception(f"Error generating CV summary: {str(e)}")