import re
from PIL import Image
import torch
from paddleocr import PaddleOCR
import easyocr
import pytesseract
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM


class OCREngine:

 
    def __init__(self, embedding_model="deepseek-r1:14b", use_paddleocr:bool=True, use_easyocr:bool=True, use_pytesseract:bool=True):
        
        self.use_paddleocr = use_paddleocr
        self.use_easyocr = use_easyocr
        self.use_pytesseract = use_pytesseract
        self.model = embedding_model


    def paddle_ocr(self, path:str):

        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        results = ocr.ocr(path, cls=True)
        output = ""
        for line in results:
            for (_, text_info) in line:
                text, _ = text_info
                output += (text + '\n')

        return output.strip('\n')
    

    def easy_ocr(self, path):

        reader = easyocr.Reader(['en'])
        results = reader.readtext(path)
        output = ""
        for _, text, _ in results:
            output += (text + '\n')

        return output.strip('\n')
    

    def tesseract_ocr(self, path):

        img = Image.open(path).convert('RGB')
        output = pytesseract.image_to_string(img, lang='eng')

        return output.strip("\n")
    

    def get_llm_output(self, paddle_text:str, easy_text:str, tesseract_text:str):

        with open('prompt_template.txt', 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        if self.use_paddleocr:
            prompt_template += f"\n\n\nPaddle OCR Output:\n{paddle_text}"
        if self.use_easyocr:
            prompt_template += f"\n\n\nEasy OCR Output:\n{easy_text}" 
        if self.use_pytesseract:
            prompt_template += f"\n\n\nTesseract OCR Output:\n  "    
        prompt_template += "\n\n\nNow generate the final quote and explanation as described above."

        llm = OllamaLLM(model=self.model)
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        response = chain.invoke({
            "paddle_text" : paddle_text,
            "easy_text" : easy_text,
            "tesseract_text" : tesseract_text
        })
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip("\n")

        return response
        
    
    def get_text(self, path:str):

        paddle_text = ""
        easy_text = ""
        tesseract_text = "" 

        if self.use_paddleocr:
            paddle_text = self.paddle_ocr(path)
            paddle_text = re.sub(r'[{}]|[^\x20-\x7E\n\t]', '', paddle_text)
        if self.use_easyocr:
            easy_text = self.easy_ocr(path)
            easy_text = re.sub(r'[{}]|[^\x20-\x7E\n\t]', '', easy_text)
        if self.use_pytesseract:
            tesseract_text = self.tesseract_ocr(path)
            tesseract_text = re.sub(r'[{}]|[^\x20-\x7E\n\t]', '', tesseract_text)
             
        return self.get_llm_output(paddle_text, easy_text, tesseract_text)
        
