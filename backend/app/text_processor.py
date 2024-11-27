from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.stem import PorterStemmer
from langdetect import detect, LangDetectException
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import string
from typing import List, Set
class TextProcessor:
    def __init__(self):
        self.stop_words_id: Set[str] = set(stopwords.words('indonesian'))
        self.stop_words_eng: Set[str] = set(stopwords.words('english'))
        stemmer_factory = StemmerFactory()
        self.stemmer_id = stemmer_factory.create_stemmer()
        self.stemmer_eng = PorterStemmer()

    def clean_text(self, text: str) -> str:
        text = self._remove_special_characters(text)
        lang = self._detect_language(text)
        return self._apply_text_transformations(text, lang)

    def _remove_special_characters(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\x00-\x7f]', '', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[{}]'.format(string.punctuation), '', text)
        return text.lower()

    def _detect_language(self, text: str) -> str:
        try:
            return detect(text)
        except LangDetectException:
            return 'en'

    def _apply_text_transformations(self, text: str, lang: str) -> str:
        stop_words = self.stop_words_eng if lang != 'id' else self.stop_words_id
        stemmer = self.stemmer_eng if lang != 'id' else self.stemmer_id
        
        tokens = word_tokenize(text)
        tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
        return ' '.join(tokens)
