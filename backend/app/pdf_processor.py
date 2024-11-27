import PyPDF2
class PDFProcessor:
    @staticmethod
    def extract_text(pdf_path: str) -> str:
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_path)
            return " ".join(page.extract_text() for page in pdf_reader.pages).strip()
        except Exception as e:
            raise Exception(f"PDF extraction failed: {str(e)}")