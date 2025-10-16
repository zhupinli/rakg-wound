from src.textPrcess import TextProcessor
from src.kgAgent import NER_Agent
import json
import os
import logging

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

def convert_to_valid_json(data):
    def format_value(obj):
        """Recursively handle single quote issues in all values"""
        if isinstance(obj, dict):
            return {k: format_value(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [format_value(item) for item in obj]
        if isinstance(obj, str):
            # Convert single quotes to double quotes (choose whether to keep based on requirements)
            return obj.replace("'", '"')
        return obj

    # Process the entire data structure
    processed_data = format_value(data)
    # Convert to strict JSON format
    return json.dumps(processed_data, 
                    indent=2, 
                    ensure_ascii=False,
                    separators=(',', ': '))

def process_all_topics(json_path, output_dir):
    # Load JSON file
    with open(json_path, 'r', encoding='utf-8') as file:
        topics = json.load(file)

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize related tools
    ner_agent = NER_Agent(logger=logger)

    start_index = 1
    # Iterate through each topic
    for idx, topic_data in enumerate(topics, start=1):
        if idx < start_index:
            continue  # Skip previous entries
        try:
            logger.info(f"Processing topic {idx}/{len(topics)}: {topic_data['topic']}")

            # Get text and topic
            text = topic_data['content']
            topic = topic_data['topic']

            # Split text
            processor = TextProcessor(text, topic)
            text_split = processor.process()

            # Extract NER and knowledge graph
            # Initial NER
            logger.info("Starting NER extraction...")
            if not os.path.exists(os.path.join(output_dir, "ner_data")):
                os.makedirs(os.path.join(output_dir, "ner_data"))
            # ner_result = ner_agent.extract_from_text_multiply(
            #     text_split['sentences'], 
            #     text_split['sentence_to_id'],
            #     output_file=os.path.join(output_dir, "ner_data", f"output_text_ner_{idx}.jsonl")
            # ) # origin
            # ner_result = ner_agent.extract_from_text_multiply(
            #     text_split['sentences'], 
            #     text_split['sentence_to_id'],
            #     output_file=os.path.join(output_dir, "ner_data", f"output_text_ner.jsonl")
            # ) # ner_result before entity disambiguation
            # logger.info("Finish NER extraction.\n")
            # with open(os.path.join(output_dir, "ner_data", f"output_ner_result.jsonl"), 'w', encoding='utf-8') as f:
            #     json.dump(ner_result, f, ensure_ascii=False, indent=4)

            # directly read ner_result
            with open(os.path.join(output_dir, "ner_data", f"output_ner_result.jsonl"), 'r', encoding='utf-8') as f:
                ner_result = json.load(f)
                logger.info("Directly read NER extraction.\n")

            # sim = ner_agent.similartiy_result(ner_result)
            # logger.info("Finish similarity calculation.\n")
            # with open(os.path.join(output_dir, "ner_data", f"ner_similarity_result_{idx}.jsonl"), 'w', encoding='utf-8') as f:
            #     json.dump(sim, f, ensure_ascii=False, indent=4)
            #     logger.info("Similarity calculation result saved.\n")

            # directly read similarity data
            with open(os.path.join(output_dir, "ner_data", f"ner_similarity_result_1.jsonl"), 'r', encoding='utf-8') as f:
                sim = json.load(f)
                logger.info("Directly read similarity calculation.\n")
            
            # NER with entity disambiguation
            if not os.path.exists(os.path.join(output_dir, "rel_data")):
                os.makedirs(os.path.join(output_dir, "rel_data"))
            entity_list_process = ner_agent.entity_Disambiguation(ner_result, sim)
            logger.info("Finish entity disambiguation.\n")
            # kg_result = ner_agent.get_target_kg_all(
            #     entity_list_process, 
            #     text_split['id_to_sentence'],
            #     text_split['sentences'],
            #     text_split['sentence_to_id'],
            #     text_split['vectors'],
            #     output_file=os.path.join(output_dir, "rel_data", f"output_kg_{idx}.jsonl")
            # ) # origin
            kg_result = ner_agent.get_target_kg_all(
                entity_list_process, 
                text_split['id_to_sentence'],
                text_split['sentences'],
                text_split['sentence_to_id'],
                text_split['vectors'],
                output_file=os.path.join(output_dir, "rel_data", f"output_kg.jsonl")
            )
            logger.info("Finish generating kg_result.\n")
            # print(kg_result)

            converted_kg = ner_agent.convert_knowledge_graph(kg_result)
            kg_json = convert_to_valid_json(converted_kg)
            logger.info("Finish converting valid json.\n")
            # print(kg_json)

            # Save to file
            output_path = os.path.join(output_dir, 'kg', f"{idx}.json")
            with open(output_path, 'w', encoding='utf-8') as outfile:
                outfile.write(kg_json)

            logger.info(f"Saved KG for topic {topic_data['topic']} to {output_path}")

        except Exception as e:
            logger.error(f"Error generating knowledge graph for entry {idx}: {str(e)}")
        

# Example call
if __name__ == "__main__":
    # json_path = "data/raw/MINE.json"  # Replace with your JSON file path

    # json_path = "data/raw/HP.json"  # Replace with your JSON file path
    # json_path = "result/burn.json"  # Replace with your JSON file path
    
    # json_path = "result/第六章_创伤_黄家驷外科学/wound.json"
    # output_dir = "result/RAKG_graph_wound"  # Output directory
    
    # json_path = "result/第六章_创伤_黄家驷外科学/wound.json"
    # output_dir = "result/RAKG_graph_wound"  # Output directory
    
    
    # input_json_file = "result/第七章_烧伤、点损伤、冷伤、咬蛰伤_黄家驷外科学/section_7.json"
    # output_file = "result/RAKG_graph_section_7"
    
    # input_json_file = "result/第四章_伤口护理_伤口护理学/第四章_伤口护理_伤口护理学.json"
    # output_file = "result/RAKG_graph_wound_care_chapter_4"

    ## test downtime
    input_json_file = "data/MINE_1.json"
    output_file = "result/RAKG_graph_mine_test"
    
    process_all_topics(input_json_file, output_file)