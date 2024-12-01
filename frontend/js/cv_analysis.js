function resumeAnalyzer() {
    return {
        jobDescription: '',
        tags: '',
        selectedFile: null,
        analysisResult: null,
        processing: false,

        handleFileSelect(event) {
            const file = event.target.files[0];
            if (file && file.size <= 10 * 1024 * 1024) {
                this.selectedFile = file;
            } else {
                alert('File size exceeds 10MB. Please upload a smaller file.');
                this.selectedFile = null;
            }
        },

        async analyzeResume() {
            if (!this.selectedFile || !this.jobDescription.trim() || !this.tags.trim()) {
                alert('Please fill in all required fields');
                return;
            }

            this.processing = true;

            const formData = new FormData();
            formData.append('file', this.selectedFile);
            formData.append('job_description', this.jobDescription);
            formData.append('tags', this.tags);

            try {
                const response = await fetch('http://127.0.0.1:8000/analyze-cv/', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error('Analysis failed');
                }

                this.analysisResult = await response.json();
            } catch (error) {
                console.error('Error:', error);
                alert('Failed to analyze resume. Please try again.');
            } finally {
                this.processing = false;
            }
        }
    }
}