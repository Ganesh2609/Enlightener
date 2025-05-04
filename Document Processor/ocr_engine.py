import re
from PIL import Image
import torch
from paddleocr import PaddleOCR
import easyocr
import pytesseract
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM


class OCREngine:
    """
    Enhanced OCR Engine that processes image files using multiple OCR systems
    and an LLM to create structured outputs optimized for vector search.
    """

    def __init__(self, embedding_model="deepseek-r1:14b", use_paddleocr=True, use_easyocr=True, use_pytesseract=True):
        """
        Initialize the OCR Engine with selected OCR systems.
        
        Args:
            embedding_model (str): The language model to use for OCR correction and structuring
            use_paddleocr (bool): Whether to use PaddleOCR
            use_easyocr (bool): Whether to use EasyOCR
            use_pytesseract (bool): Whether to use Pytesseract
        """
        self.use_paddleocr = use_paddleocr
        self.use_easyocr = use_easyocr
        self.use_pytesseract = use_pytesseract
        self.model = embedding_model

    def paddle_ocr(self, path: str):
        """
        Extract text from an image using PaddleOCR.
        
        Args:
            path (str): Path to the image file
            
        Returns:
            str: Extracted text
        """
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        results = ocr.ocr(path, cls=True)
        output = ""
        for line in results:
            for (_, text_info) in line:
                text, _ = text_info
                output += (text + '\n')

        return output.strip('\n')
    
    def easy_ocr(self, path):
        """
        Extract text from an image using EasyOCR.
        
        Args:
            path (str): Path to the image file
            
        Returns:
            str: Extracted text
        """
        reader = easyocr.Reader(['en'])
        results = reader.readtext(path)
        output = ""
        for _, text, _ in results:
            output += (text + '\n')

        return output.strip('\n')
    
    def tesseract_ocr(self, path):
        """
        Extract text from an image using Pytesseract.
        
        Args:
            path (str): Path to the image file
            
        Returns:
            str: Extracted text
        """
        img = Image.open(path).convert('RGB')
        output = pytesseract.image_to_string(img, lang='eng')

        return output.strip("\n")
    
    def get_llm_output(self, paddle_text: str, easy_text: str, tesseract_text: str):
        """
        Use LLM to correct OCR output and structure it for vector search.
        
        Args:
            paddle_text (str): Text extracted by PaddleOCR
            easy_text (str): Text extracted by EasyOCR
            tesseract_text (str): Text extracted by Pytesseract
            
        Returns:
            str: Processed and structured output
        """
        with open('prompt_template.txt', 'r', encoding='utf-8') as f:
            prompt_template = f.read()
            
        # Add OCR outputs to the prompt
        if self.use_paddleocr:
            prompt_template += f"\n\n\nPaddle OCR Output:\n{paddle_text}"
        if self.use_easyocr:
            prompt_template += f"\n\n\nEasy OCR Output:\n{easy_text}" 
        if self.use_pytesseract:
            prompt_template += f"\n\n\nTesseract OCR Output:\n{tesseract_text}"    
        
        prompt_template += "\n\n\nNow generate the final quote and explanation as described above."

        llm = OllamaLLM(model=self.model)
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        response = chain.invoke({
            "paddle_text": paddle_text,
            "easy_text": easy_text,
            "tesseract_text": tesseract_text
        })
        
        # Remove thinking process if present
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip("\n")
        
        # Verify output format
        if not self._validate_output_format(response):
            response = self._fix_output_format(response)
            
        return response
    
    def _validate_output_format(self, response: str) -> bool:
        """
        Validates that the output follows the required format.
        
        Args:
            response (str): The LLM response to validate
            
        Returns:
            bool: True if format is valid, False otherwise
        """
        required_sections = ["QUOTE:", "EXPLANATION:"]
        
        # Keywords section is also expected but might not be present in older prompts
        for section in required_sections:
            if section not in response:
                return False
                
        return True
    
    def _fix_output_format(self, response: str) -> str:
        """
        Attempts to fix the output format if validation fails.
        
        Args:
            response (str): The LLM response to fix
            
        Returns:
            str: Fixed response
        """
        # If no QUOTE section but has quotes, try to extract
        if "QUOTE:" not in response and '"' in response:
            quote_match = re.search(r'"([^"]*)"', response)
            if quote_match:
                quote = quote_match.group(1)
                response = f"QUOTE: {quote}\n\n" + response.replace(f'"{quote}"', "")
        
        # If no EXPLANATION section but has content after QUOTE
        if "QUOTE:" in response and "EXPLANATION:" not in response:
            parts = response.split("QUOTE:")
            if len(parts) > 1:
                quote_part = parts[1].strip()
                # Split on double newline if present
                content_parts = quote_part.split("\n\n", 1)
                if len(content_parts) > 1:
                    response = f"QUOTE: {content_parts[0]}\nEXPLANATION: {content_parts[1]}"
                else:
                    # If no clear separation, just split after first sentence
                    sentences = quote_part.split(". ", 1)
                    if len(sentences) > 1:
                        response = f"QUOTE: {sentences[0]}.\nEXPLANATION: {sentences[1]}"
        
        # Add KEYWORDS section if missing
        if "KEYWORDS:" not in response:
            response += "\nKEYWORDS: spirituality, wisdom, growth, peace, guidance, inner transformation"
            
        return response
    
    def get_text(self, path: str):
        """
        Process an image file and extract structured text.
        
        Args:
            path (str): Path to the image file
            
        Returns:
            str: Processed and structured output
        """
        paddle_text = ""
        easy_text = ""
        tesseract_text = "" 

        # Get text from different OCR systems
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