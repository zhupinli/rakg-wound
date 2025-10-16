import re
import os
import json
import traceback
import torch
import pdfplumber

from tqdm import tqdm
from src.llm_provider import LLMProvider
from collections import Counter
from surya.layout import LayoutPredictor
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont

from surya.layout import LayoutPredictor
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor


class pdfExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.base_name = os.path.splitext(os.path.basename(pdf_path))[0]  # Extract filename
        self.sentence_to_id = {}
        self.id_to_sentence = {}
        self.llm_provider = LLMProvider()
        self.embeddings = self.llm_provider.get_embedding_model()

    def convert_layout_to_dict(self, layout_predictions):
        """
        将 LayoutResult 对象列表转换为可 JSON 序列化的字典列表。
        """
        output_list = []
        # layout_predictions 是一个列表，每个元素对应一页的结果
        for i, page_result in enumerate(layout_predictions):
            # page_result 是一个 LayoutResult 对象
            page_dict = {
                "page_number": i + 1,
                "bboxes": [],
                # 如果需要，可以添加其他 LayoutResult 的属性
                # "image_size": page_result.image_size 
            }
            
            # 遍历该页所有的 Bbox 对象
            for bbox in page_result.bboxes:
                bbox_dict = {
                    "label": bbox.label,
                    "confidence": bbox.confidence,
                    "bbox": bbox.bbox,  # 边界框坐标 [x1, y1, x2, y2]
                    "polygon": bbox.polygon,  # 多边形坐标
                }
                page_dict["bboxes"].append(bbox_dict)
            
            output_list.append(page_dict)
        return output_list

    def draw_layout_predictions(self, image, predictions, output_path):
        """
        在图片上绘制版面识别框并保存。

        参数:
        image: 原始图片。
        predictions (list): surya layout预测结果的列表，每个元素对应一页。
        output_path (str): 绘制了识别框的图片的保存路径。
        """
        try:
            # 打开原始图片
            draw = ImageDraw.Draw(image)

            # 定义不同标签的颜色
            label_colors = {
                # 已有的标签
                "Text": "red",
                "Title": "blue",
                "Table": "green",
                "Figure": "purple",
                "SectionHeader": "cyan",  # 与下方的 Section-header 保持一致
                "Caption": "yellow",
                "PageFooter": "orange",  # 与下方的 Page-footer 保持一致

                # 根据您的列表新增和更新的标签
                "Footnote": "magenta",
                "Formula": "brown",
                "List-item": "pink",
                "Page-footer": "orange",
                "Page-header": "teal",
                "Picture": "purple",         # 与 Figure 颜色相同
                "Section-header": "cyan",
                "Form": "grey",
                "Table-of-contents": "olive",
                "Handwriting": "navy",
                "Text-inline-math": "gold"
            }

            # 加载一个字体用于显示标签，如果找不到字体文件，会使用默认字体
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()

            # 遍历这一页的所有识别框
            # 注意: predictions[0] 代表第一页的预测结果
            for i, bbox_info in enumerate(predictions):
                box = bbox_info.bbox
                label = bbox_info.label
                
                # 获取颜色，如果标签不在字典里，则使用黑色
                color = label_colors.get(label, "black")
                # 绘制矩形框
                draw.rectangle(box, outline=color, width=3)
                # 在框的左上角绘制标签文本
                text_position = (box[0] + 5, box[1] + 5)
                draw.text(text_position, f"{label}  {i}", fill=color, font=font)

            # 保存图片
            image.save(output_path, "PNG")
            print(f"layout predictions on image has been saved to: {output_path}")

        except Exception as e:
            print(f"处理图片时发生错误: {e}")
            print(traceback.format_exc())
    
    def ocr_specific_area(self, detection_predictor, recognition_predictor, image, roi_bbox):
        """
        对图片中的一个指定矩形区域（Region of Interest）进行 OCR。

        参数:
        image_path (img obj): 原始图片
        roi_bbox (list or tuple): 一个包含四个整数的列表或元组，
                                格式为 [x_min, y_min, x_max, y_max]，
                                定义了要识别的区域。
        """

        # --- 3. 裁剪出指定的区域 ---
        print(f"正在从原始图片中裁剪区域: {roi_bbox}...")
        cropped_image = image.crop(roi_bbox)

        # --- 4. 对裁剪出的图片进行 OCR ---
        print("正在对裁剪区域进行文本检测和识别...")
        # 首先，检测文本行位置
        det_predictions = detection_predictor([cropped_image])
        text_polygons = [line.polygon for line in det_predictions[0].bboxes]
        
        # 识别文本内容
        rec_predictions = recognition_predictor(images=[cropped_image], polygons=[text_polygons])

        # --- 5. 提取并返回结果 ---
        if not rec_predictions or not rec_predictions[0].text_lines:
            print("没有识别到任何文本行。")
            return ""
        
        recognized_lines = rec_predictions[0].text_lines
        print("\n--- 指定区域内的识别结果 ---")
        final_text_block = []
        for line in recognized_lines:
            print(line.text)
            final_text_block.append(line.text)
        # 将所有行合并成一个文本块
        return "\n".join(final_text_block)
    
    def examine_layout_prediction_order(self, images, layout_predictions, output_folder):

        # layout_predictor = LayoutPredictor()
        # images = convert_from_path(self.pdf_path)
        # layout_predictions = layout_predictor(images)

        if not os.path.exists(os.path.join(output_folder, "img")):
            os.makedirs(os.path.join(output_folder, "img"))
        for i, (image, layout) in enumerate(zip(images, layout_predictions)):
            self.draw_layout_predictions(image, layout, os.path.join(output_folder, "layout_img", f"page_{i}_layout_order.png"))
        
        print(f"layout predictions have been saved to {os.path.join(output_folder, 'img')}.")

    def sort_layout_reading_order(self, layout_result, page_width):
        """
        直接按阈值把 box 分到左右两栏，然后分别按从上到下排序，最后拼接成阅读顺序。
        
        Args:
        - layout_result: Surya 返回的 LayoutResult 对象
        - page_width: 页面宽度，用于把 box 分到左/右栏
        
        Returns:
        - reading_order: 按阅读顺序排好序的 LayoutBox 列表
        """

        boxes = layout_result.bboxes
        
        # 页面中线 x 坐标
        mid_x = page_width / 2
        
        # 分左右栏
        left_col  = []
        right_col = []
        for box in boxes:
            cx = box.center[0]
            if cx < mid_x:
                left_col.append(box)
            else:
                right_col.append(box)
        
        # 每列内部按 y0 排序（y0 越小越靠上）
        left_col_sorted  = sorted(left_col,  key=lambda b: b.bbox[1])
        right_col_sorted = sorted(right_col, key=lambda b: b.bbox[1])
        
        # 拼接：先左栏再右栏
        reading_order = left_col_sorted + right_col_sorted
        return reading_order
    

    def extract_surya(self, output_folder):

        # --- 1. 初始化 Predictor ---
        layout_predictor = LayoutPredictor()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        detection_predictor = DetectionPredictor(device=device)
        recognition_predictor = RecognitionPredictor(device=device)
        img_id = 1

        # convert pdf to images
        images = convert_from_path(self.pdf_path)
        layout_predictions = layout_predictor(images)

        # convert layout predictions to human reading order
        layout_results_reading_order = []
        for layout_result, image in zip(layout_predictions, images):
            # 获取页面宽度
            page_width = image.size[0]
            result_reading_order = self.sort_layout_reading_order(layout_result, page_width)
            layout_results_reading_order.append(result_reading_order)


        # --- 2. 对每一页进行 OCR ---    
        ocr_result_list = []
        pdf_result_dict_list = []    
        for i, (image, layout) in enumerate(zip(images, layout_results_reading_order)):

            # OCR specific area
            ocr_result = []
            pdf_result_dict = []
            current_section = None

            if not os.path.exists(os.path.join(output_folder, "img")):
                os.makedirs(os.path.join(output_folder, "img"))
            if not os.path.exists(os.path.join(output_folder, "layout_img")):
                os.makedirs(os.path.join(output_folder, "layout_img"))

            for bbox in layout:
                
                label = bbox.label
                bbox = bbox.bbox

                if label == "SectionHeader":
                    # structual pdf result
                    # 扫到新章节
                    header_text = self.ocr_specific_area(detection_predictor, recognition_predictor, image, bbox)
                    ocr_result.append(header_text)
                    # 新建一个节的 dict
                    current_section = {
                        "section": header_text,
                        "content": []
                    }
                    pdf_result_dict.append(current_section)

                elif label in ["Text", "Title", "Caption"]:
                    # 只对特定标签的区域进行 OCR
                    recognized_text = self.ocr_specific_area(detection_predictor, recognition_predictor, image, bbox)
                    ocr_result.append(recognized_text)
                    print(f"识别区域 {label} 的文本: {recognized_text}")

                    # structual pdf result
                    content_item = {
                        "text": recognized_text,
                        "images": []
                    }
                    if current_section is None:
                        # 如果还没遇到 SectionHeader，则先建个默认节
                        current_section = {"section": "", "content": []}
                        pdf_result_dict.append(current_section)
                    current_section["content"].append(content_item)

                elif label in ["Picture", "Figure", "Table"]:
                    # 对图片区域进行 OCR
                    cropped_image = image.crop(bbox)
                    cropped_image_output_path = os.path.join(output_folder, "img", f"cropped_{label}_page_{img_id}.png")
                    cropped_image.save(cropped_image_output_path)
                    img_id += 1
                    # 附加到当前节里最后一个 content
                    if current_section and current_section["content"]:
                        current_section["content"][-1]["images"].append(cropped_image_output_path)
                    # 否则忽略，或根据需要新建一条
            
            ocr_result_list.extend(ocr_result)
            pdf_result_dict_list.extend(pdf_result_dict)

            if not os.path.exists(os.path.join(output_folder, "pdf_result")):
                os.makedirs(os.path.join(output_folder, "pdf_result"))
            with open(os.path.join(output_folder, "pdf_result", "ocr_result.txt"), "a", encoding="utf-8") as f:
                for text in ocr_result_list:
                    f.write(text + "\n")
            with open(os.path.join(output_folder, "pdf_result", "pdf_result_structured.json"), "a", encoding="utf-8") as f:
                json.dump(pdf_result_dict_list, f, ensure_ascii=False, indent=2)

            # draw bbox on page
            # self.draw_layout_predictions(image, layout, os.path.join(output_folder, "layout_img", f"page_{i}_layout.png"))


        # 绘制阅读顺序的布局预测结果
        self.examine_layout_prediction_order(images, layout_results_reading_order, output_folder)

        # # save results
        # if not os.path.exists(os.path.join(output_folder, "pdf_result")):
        #     os.makedirs(os.path.join(output_folder, "pdf_result"))
        # with open(os.path.join(output_folder, "pdf_result", "ocr_result.txt"), "w", encoding="utf-8") as f:
        #     for text in ocr_result_list:
        #         f.write(text + "\n")
        # with open(os.path.join(output_folder, "pdf_result", "pdf_result_structured.json"), "w", encoding="utf-8") as f:
        #     json.dump(pdf_result_dict_list, f, ensure_ascii=False, indent=2)
        
        return layout_predictions
    
    def extract_single_column(self, pdf_path):
        all_text = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # 提取页面的文本
                page_text = page.extract_text(x_tolerance=2)
                if page_text:
                    all_text.append(page_text.strip())
                    
        return "\n".join(all_text)
    
    def extract_double_column(self, pdf_path):
        all_text = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                x0, y0, x1, y1 = page.bbox  # 实际页面边界
                mid_x = (x0 + x1) / 2       # 页面中线（基于实际 bbox）

                # 切左栏和右栏
                left_bbox = (x0, y0, mid_x, y1)
                right_bbox = (mid_x, y0, x1, y1)

                # 分别提取左右栏的文本
                left_text = page.within_bbox(left_bbox).extract_text(x_tolerance=2)
                right_text = page.within_bbox(right_bbox).extract_text(x_tolerance=2)

                # 合并栏内容（注意：左栏在右栏前）
                page_text = ""
                if left_text:
                    page_text += left_text.strip() + "\n"
                if right_text:
                    page_text += right_text.strip()

                if page_text.strip():
                    all_text.append(page_text.strip())
                    
        return "\n".join(all_text)
    
    ## Split text into segments and return a list
    def split_sentences(self, text):
        """Support Chinese and English sentence segmentation (handling common abbreviations)"""
        pattern = re.compile(r'(?<!\b[A-Za-z]\.)(?<=[.!?。！？])\s+')
        sentences = [s.strip() for s in pattern.split(text) if s.strip()]
        return sentences
    
    def generate_id(self, index):
        """Generate ID according to requirements"""
        return f"{self.base_name}{index+1}"
    
    def process(self):
        text = self.extract_single_column(self.pdf_path)
        # text = self.extract_double_column(self.pdf_path)
        
        if not os.path.exists(os.path.join("result", self.base_name)):
            os.makedirs(os.path.join("result", self.base_name))
        
        with open(os.path.join("result", self.base_name, f"{self.base_name}_raw_text.json"), "w", encoding="utf-8") as f:
            json.dump(text, f, ensure_ascii=False, indent=4)
        print("Raw text has been saved to:", os.path.join("result", self.base_name, f"{self.base_name}_raw_text.json"))
        
        sentences = self.split_sentences(text)
        for idx, sentence in enumerate(sentences):
            sent_id = self.generate_id(idx)
            self.sentence_to_id[sentence] = sent_id
            self.id_to_sentence[sent_id] = sentence
        
        vectors = self.embeddings.embed_documents(sentences)

        pdf_process_result = {
            "sentences": sentences,
            "vectors": vectors,
            "sentence_to_id": self.sentence_to_id,
            "id_to_sentence": self.id_to_sentence
        }
        with open(os.path.join("result", self.base_name, f"{self.base_name}_process_result.json"), "w", encoding="utf-8") as f:
            json.dump(pdf_process_result, f, ensure_ascii=False, indent=4)
        print("Processed result has been saved to:", os.path.join("result", self.base_name, f"{self.base_name}_process_result.json"))
        
        joined = '\n\n'.join(pdf_process_result['sentences'])
        processed_data = [
            {"topic": f"{os.path.basename(self.pdf_path).split('.')[0]}", 
             "content": f"```{joined}```"}
        ]
        with open(os.path.join("result", self.base_name, f"{self.base_name}.json"), "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=4)  
        
        
        return pdf_process_result
       

if __name__=='__main__':

    # pdf_path = "pdf/第七章_第一节_1,2_烧伤_黄家驷外科学.pdf"
    # pdf_path = "pdf/第七章_第一节_烧伤_黄家驷外科学.pdf"
    # pdf_path = "pdf/第六章_创伤_黄家驷外科学.pdf"
    # pdf_path = "pdf/第七章_烧伤、点损伤、冷伤、咬蛰伤_黄家驷外科学.pdf"
    pdf_path = "pdf/第四章_伤口护理_伤口护理学.pdf"

    # output_folder = "result/layout_detection_result_section_seven_test"
    # output_folder = "result/layout_detection_result_section_seven"
    # output_folder = "result/layout_detection_result_section_seven"

    # test extractor
    # extractor = pdfExtractor(pdf_path)
    # result = extractor.extract_surya(output_folder)
    # print(result)
    
    extractor = pdfExtractor(pdf_path)
    extractor.process()
