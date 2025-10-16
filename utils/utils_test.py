import json 
import layoutparser as lp
import pdfplumber
import os
import traceback
import re
import logging

from tqdm import tqdm
from pdf2image import convert_from_path
from src.construct.RAKG_wound import convert_to_valid_json
from src.kgAgent import NER_Agent
from src.textPrcess import TextProcessor

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("test_mine.log", encoding="utf-8"),
        logging.StreamHandler() 
    ]
)
logger = logging.getLogger(__name__)

def process_raw_json(input_json_file, output_file):
    '''
    convert raw json file to a format can be used by rakg-construct
    '''
    data = None

    with open(input_json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    joined = '\n\n'.join(data['sentences'])

    with open(output_file, 'w', encoding='utf-8') as file:
        output_data = [
            {"topic": f"{os.path.basename(input_json_file).split('.')[0]}", 
             "content": f"```{joined}```"}
        ]
        json.dump(output_data, file, ensure_ascii=False, indent=4)
    print(f"Processed data saved to {output_file}")


def layout_detection(pdf_path):
    # 加载布局检测模型
    model = lp.AutoLayoutModel('lp://efficientdet/PubLayNet')
    pages = convert_from_path(pdf_path, dpi=150)

    layout_result = []

    for i, pil_image in enumerate(tqdm(pages), start=1):
        pil_image = pil_image.convert("RGB")
        layout = model.detect(pil_image)
        layout_result.append(layout)

        print(f"Page {i} blocks:")
        for block in layout:
            print(block)

        annotated = lp.draw_box(pil_image, layout, box_width=3, show_element_type=True)
        # annotated.show()
        annotated.save(f'result/img/img_{i}.png')

def convert_encode():
    input_file = 'result/rakg_graph_v1/temp_ner_output.jsonl'
    output_file = 'result/rakg_graph_v1.json/temp_ner_output_processed.jsonl'
    data = None

    with open(input_file, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]

    with open(output_file, 'w', encoding='utf-8') as file:
        for item in data:
            item['text'] = item['text'].encode('utf-8').decode('utf-8')
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')


def process_ner_and_rel():
    ner_data_file = 'data/processed/llmasjudge/ner_data/output_text_ner_1.jsonl'
    rel_data_file = 'data/processed/llmasjudge/rel_data/output_kg_1.jsonl'
    output_folder = 'result/rakg_graph_v1'
    ner_data = []
    rel_data = []

    with open(ner_data_file, 'r', encoding='utf-8') as file:
        for line in file:
            ner_data.append(json.loads(line))
    
    with open(rel_data_file, 'r', encoding='utf-8') as file:
        for line in file:
            rel_data.append(json.loads(line))
    
    ner_data_new = ner_data[37:]
    rel_data_new = rel_data[17:]

    with open(os.path.join(output_folder, 'output_text_ner_1.jsonl'), 'w', encoding='utf-8') as file:
        json.dump(ner_data_new, file, ensure_ascii=False)

    with open(os.path.join(output_folder, 'output_kg_1.jsonl'), 'w', encoding='utf-8') as file:
        json.dump(rel_data_new, file, ensure_ascii=False)


def generate_kg_from_ner_and_rel(rel_data_dir, output_dir, out_file_name):
    
    ## generate kg
    ner_agent = NER_Agent()
    raw_list = []
    
    try:
        # 如果是单行 JSONL 格式
        with open(rel_data_dir, "r", encoding="utf-8") as f:
            raw_list = [json.loads(line) for line in f]
    except json.JSONDecodeError:
        # 如果是标准 JSON 格式
        with open(rel_data_dir, "r", encoding="utf-8") as f:
            raw_list = json.load(f)

    def extract_entity_id(entity):
        # 取第一个 chunkid 片段作为 id
        return entity["chunkid"].split(";;;")[0]

    kg_result = {
        extract_entity_id(item["entity"]): item["kg"]
        for item in raw_list
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        converted_kg = ner_agent.convert_knowledge_graph(kg_result)
        kg_json = convert_to_valid_json(converted_kg)
        print("kg_json(modified_normalize) generated successfully.")
        # print(kg_json)
        # Save to file
        output_path = os.path.join(output_dir, out_file_name) # TODO test normalization
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(kg_json)
    except Exception as e:
        print(f"Error generating knowledge graph: {str(e)}")
        traceback.print_exc()  # 打印完整的异常堆栈信息


def read_kg_result_json():
    kg_result_file = 'result/processed/RAKG_graph_v1/rakg_kg.json'
    output_folder = 'result/rakg_graph_v1'
    raw = None
    with open(kg_result_file, 'r', encoding='utf-8') as file:
        raw = json.load(file)
    kg_result = json.loads(raw)
    print("KG Result:\n\n", kg_result)

    with open(os.path.join(output_folder, 'kg_result.json'), 'w', encoding='utf-8') as file:
        json.dump(kg_result, file, ensure_ascii=False, indent=4)


def read_example():

    kg_result_file = 'data/processed/RAKG_graph_v2/1.json'
    output_folder = 'result/rakg_graph_v1'
    raw = None
    with open(kg_result_file, 'r', encoding='utf-8') as file:
        raw = json.load(file)
    kg_result = json.loads(raw)
    print("KG Result:\n\n", kg_result)

    with open(os.path.join(output_folder, 'kg_result_examples.json'), 'w', encoding='utf-8') as file:
        json.dump(kg_result, file, ensure_ascii=False, indent=4)
    
def read_entity_relation(output_dir):
    ner_data_file = os.path.join(output_dir, 'ner_data', 'output_text_ner_1.jsonl')
    ner_data_file_convert = os.path.join(output_dir, 'ner_data', 'output_text_ner_1_convert.jsonl')
    rel_data_file = os.path.join(output_dir, 'rel_data', 'output_kg_1.jsonl')
    rel_data_file_convert = os.path.join(output_dir, 'rel_data', 'output_kg_1_convert.jsonl')
    output_dir = os.path.join(output_dir, 'kg_result', 'kg_result.json')
    
    ner_data = []
    rel_data = []
    
    with open(ner_data_file, 'r', encoding='utf-8') as file:
        for line in file:
            ner_data.append(json.loads(line))
        with open(ner_data_file_convert, 'w', encoding='utf-8') as convert_file:
            json.dump(ner_data, convert_file, ensure_ascii=False, indent=4)
    with open(rel_data_file, 'r', encoding='utf-8') as file: 
        for line in file:
            rel_data.append(json.loads(line))
        with open(rel_data_file_convert, 'w', encoding='utf-8') as convert_file:
            json.dump(rel_data, convert_file, ensure_ascii=False, indent=4)
    
def debug_gen_kg():
    
    input_file_dir = "data/MINE_1.json"
    ner_result_dir = "result/RAKG_graph_mine_test/ner_data/output_text_ner_1.jsonl"
    similarity_result_dir = "result/RAKG_graph_mine_test/ner_data/ner_similarity_result_1.jsonl"
    output_dir = "result/RAKG_graph_mine_test/kg"

    ner_agent = NER_Agent()
    
    with open(input_file_dir, 'r', encoding='utf-8') as file:
        topics = json.load(file)[0]
    
    logger.info(f"Processing topic: {topics['topic']}")
    
    text = topics["content"]
    topic = topics["topic"]
    processor = TextProcessor(text, topic)
    text_split = processor.process()

    # entity disambiguation
    ner_result = []
    similarity_result = []

    with open(ner_result_dir, 'r', encoding='utf-8') as f:
        for line in f:
            ner_result.append(json.loads(line))
    with open(similarity_result_dir, 'r', encoding='utf-8') as f:
        similarity_result = json.load(f)

    logger.info("Starting entity disambiguation...")
    entity_list_process = ner_agent.entity_Disambiguation(ner_result, similarity_result)
    logger.info("Finish entity disambiguation.\n")
    
    logger.info("Starting merging knowledge graph...")    
    kg_result = ner_agent.get_target_kg_all(
        entity_list_process, 
        text_split['id_to_sentence'],
        text_split['sentences'],
        text_split['sentence_to_id'],
        text_split['vectors'],
        output_file=os.path.join(output_dir, "rel_data", f"output_kg_mine_test.jsonl")
    )
    logger.info("Finish merging knowledge graph.\n")
    
    converted_kg = ner_agent.convert_knowledge_graph(kg_result)
    kg_json = convert_to_valid_json(converted_kg)
    logger.info("Finish converting valid json.\n")
    print(kg_json)
    



if __name__ == "__main__":

    # pdf_path = "pdf/第七章_第一节_烧伤_黄家驷外科学.pdf"

    # convert raw json file to a format can be used by rakg-construct
    # input_json_file = "result/第七章_烧伤、点损伤、冷伤、咬蛰伤_黄家驷外科学/第七章_烧伤、点损伤、冷伤、咬蛰伤_黄家驷外科学_process_result.json"
    # output_file = "result/第七章_烧伤、点损伤、冷伤、咬蛰伤_黄家驷外科学/section_7.json"
    # process_raw_json(input_json_file, output_file)

    # convert_encode()

    # process_ner_and_rel()

    # generate_kg_from_ner_and_rel()

    # read_kg_result_json()

    # read_example()
    
    # TODO ner_data json, rel_data json is encoded in unicode, show as garbled text
    # output_dir = 'result/RAKG_graph_wound'
    # read_entity_relation(output_dir)
    
    ### test kgAgent.convert_knowledge_graph()
    out_file_name = 'kg_result_modified_normalize_debug_v2.json'
    # wound chapter 7
    # rel_data_dir = 'result/RAKG_graph_wound_chapter_7/rel_data/output_kg_1.jsonl'
    # output_dir = 'result/RAKG_graph_wound_chapter_7/kg_data'
    # wound chapter 6
    rel_data_dir = 'result/RAKG_graph_wound_chapter_6/rel_data/output_kg_1_convert.jsonl'
    output_dir = 'result/RAKG_graph_wound_chapter_6/kg_data'
    # wound care chapter 4
    # rel_data_dir = 'result/RAKG_graph_wound_care_chapter_4/rel_data/output_kg_1.jsonl'
    # output_dir = 'result/RAKG_graph_wound_care_chapter_4/kg'
    generate_kg_from_ner_and_rel(rel_data_dir, output_dir, out_file_name)

    # debug_gen_kg()