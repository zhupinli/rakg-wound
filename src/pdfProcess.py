import re
import os
import pdfplumber
import json

from tqdm import tqdm
from .llm_provider import LLMProvider
from collections import Counter


class pdfProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.base_name = os.path.splitext(os.path.basename(pdf_path))[0]  # Extract filename
        self.sentence_to_id = {}
        self.id_to_sentence = {}
        self.llm_provider = LLMProvider()
        self.embeddings = self.llm_provider.get_embedding_model()
    

    def extract_text_from_pdf(self, pdf_path):

        full_text = []  
        with pdfplumber.open(pdf_path) as pdf:
            for page in tqdm(pdf.pages):
                text = page.extract_text()
                if text:
                    full_text.append(text)  

        return "\n".join(full_text)
    
    def extract_double_column(self, pdf_path):
        all_pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for p in tqdm(pdf.pages):
                W, H = p.bbox[2], p.bbox[3]
                # 去掉页眉 (0-50 pt) 和页脚 (H-50, H)
                want_bbox = (0, 50, W, H - 50)
                content = p.crop(want_bbox)

                # 左栏 / 右栏
                mid = W * 0.5       
                left  = content.crop((0,    50, mid, content.height))
                right = content.crop((mid,  50, W, content.height))

                page_text = []
                # `x_tolerance=1` 能减少“字串跨栏合并”的机会
                for col in (left, right):
                    txt = col.extract_text(x_tolerance=1) or ""
                if txt.strip():
                    page_text.append(txt.strip())
                all_pages.append("\n".join(page_text))
        return "\n\n".join(all_pages)

    
    ## Split text into segments and return a list
    def split_sentences(self, text):
        """Support Chinese and English sentence segmentation (handling common abbreviations)"""
        pattern = re.compile(r'(?<!\b[A-Za-z]\.)(?<=[.!?。！？])\s+')
        sentences = [s.strip() for s in pattern.split(text) if s.strip()]
        return sentences
    
    def generate_id(self, index):
        """Generate ID according to requirements"""
        return f"{self.base_name}{index+1}"
    
    def process(self, output_path="result"):
        text = self.extract_double_column(self.pdf_path)

        sentences = self.split_sentences(text)
        for idx, sent in enumerate(sentences):
            sent_id = self.generate_id(idx)
            self.sentence_to_id[sent] = sent_id
            self.id_to_sentence[sent_id] = sent
        
        vectors = self.embeddings.embed_documents(sentences)

        pdf_process_result = {
            "sentences": sentences,
            "vectors": vectors,
            "sentence_to_id": self.sentence_to_id,
            "id_to_sentence": self.id_to_sentence
        }

        if not os.path.exists(os.path.join(output_path, f"{self.base_name}")):
            os.makedirs(os.path.join(output_path, f"{self.base_name}"))

        with open(os.path.join(output_path, f"{self.base_name}", "raw_text.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Extracted text from {self.pdf_path} and saved to {os.path.join(output_path, f'{self.base_name}', 'raw_text.txt')}")

        with open(os.path.join(output_path, f"{self.base_name}", "processed_result.json"), "w", encoding="utf-8") as f:
            json.dump(pdf_process_result, f, ensure_ascii=False, indent=4)
        print(f"Processed sentences and vectors saved to {os.path.join(output_path, f'{self.base_name}', 'processed_result.json')}")

        return pdf_process_result         

if __name__=='__main__':
    # Usage example
    # pdf_path = "pdf/第七章_第一节_烧伤_黄家驷外科学.pdf"
    pdf_path = "pdf/第六章_创伤_黄家驷外科学.pdf"

    output_path = "result"
    processor = pdfProcessor(pdf_path)
    # processor = pdfProcessor("pdf/第六章_创伤_黄家驷外科学.pdf")
    
    # 抽取文本
    result = processor.process(output_path)
    
