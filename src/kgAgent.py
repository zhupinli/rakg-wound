from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from src.prompt import text2entity_en
from src.prompt import extract_entiry_centric_kg_en_v2
from src.prompt import judge_sim_entity_en
from itertools import combinations
from typing import Dict, Any
import re
import json
import ast
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.llm_provider import LLMProvider
import logging
import time
from utils import normalize_attributes, normalize_attributes_dict, normalize_attributes_dict_origin, normalize_relationships
from tenacity import retry, stop_after_attempt, wait_fixed, RetryCallState

class NER_Agent():
    def __init__(self, logger=None):
        self.llm_provider = LLMProvider()
        self.model = self.llm_provider.get_llm()
        self.similarity_model = self.llm_provider.get_similarity_model()
        self.embeddings = self.llm_provider.get_embedding_model()
        self.logger = logger if logger else logging.getLogger(__name__)

    ## Add chunkid attribute
    def add_chunkid(self, ner_result, chunkid):
        new_ner_result = {}
        for entity_key, entity_value in ner_result.items():
            entity_value["chunkid"] = chunkid
            new_ner_result[entity_key] = entity_value
        return new_ner_result
    
    def extract_from_text_single(self, text_single, output_file):
        prompt = ChatPromptTemplate.from_template(text2entity_en)
        chain = prompt | self.model
        try:
            result_json = self.call_llm_with_timeout(chain, {"text": text_single})
        except Exception as e:
            self.logger.warning(f"LLM inoke failure after all retries: {e}")
            result_json = {}
        
        # result = chain.invoke({"text": text_single})
        # # if hasattr(result, 'content'):
        #     result_json = json.loads(result.content)
        # else:
        #     result_json = json.loads(result)
        
        # Store text_single and result_json in a jsonl file

        with open(output_file, 'a') as f:
            combined_data = {
                "text": text_single,
                "entities": result_json
            }
            f.write(json.dumps(combined_data, ensure_ascii=False) + '\n')
        
        return result_json
    
    def rewrite(self, ner_result, entity_num):
        new_entities = {}
        # Process in original dictionary key order, extract numbers after entity and renumber
        for idx, (old_key, value) in enumerate(ner_result.items(), start=1):
            new_key = f"entity{entity_num + idx - 1}"
            new_entities[new_key] = value
        return new_entities
    
    ## Implement named entity recognition for the entire text and add chunkid field to each entity
    def extract_from_text_multiply(self, text_list, sent_to_id, output_file):
        ner_result_for_all = {}
        entity_num = 1
        for text in text_list:
            ner_result = self.extract_from_text_single(text, output_file)
            ## Add a check here - if ner_result has a state field, it means there's an issue with this chunk, so skip to the next iteration
            if 'State' in ner_result:
                continue
            # Get the number of entities in ner_result
            ner_result_num = len(ner_result)
            # Rewrite ner_result, entity numbering starts from entity_num, first entity is entity{entity_num}, subsequent entities increment
            ner_result = self.rewrite(ner_result, entity_num)

            entity_num += ner_result_num
            ner_result_with_chunkid = self.add_chunkid(ner_result, sent_to_id[text])
            ner_result_for_all.update(ner_result_with_chunkid)
        return ner_result_for_all
    
    
    def similarity_candidates(self, entities, threshold=0.60):
        def get_embedding_vector(text):
            result = self.embeddings.embed_documents([text])
            if isinstance(result, list) and isinstance(result[0], list):
                return result[0]
            else:
                return result
        entity_texts = {
            k: f"{v['name']} {v['type']}"
            for k, v in entities.items()
        }
        vectors = {
            k: get_embedding_vector(text)
            for k, text in entity_texts.items()
        }

        keys = list(vectors.keys())
        sim_matrix = np.zeros((len(keys), len(keys)))

        for i, j in combinations(range(len(keys)), 2):
            sim = cosine_similarity([vectors[keys[i]]], [vectors[keys[j]]])
            # print(f"Similarity between {keys[i]} and {keys[j]}: {sim}")
            sim_matrix[i][j] = sim

        candidates = [
            (keys[i], keys[j])
            for i, j in zip(*np.where(sim_matrix > threshold))
        ]
        return candidates


    def similarity_llm_single(self, entity1, entity2):
        prompt = ChatPromptTemplate.from_template(judge_sim_entity_en)
        chain = prompt | self.similarity_model
        try:
            result_json = self.call_llm_with_timeout(chain, {"entity1": str(entity1), "entity2": str(entity2)})
        except Exception as e:
            self.logger.warning(f"LLM inoke failure after all retries: {e}")
            result_json = {}
        # result = chain.invoke({"entity1": str(entity1), "entity2": str(entity2)})
        # if hasattr(result, 'content'):
        #     result_json = json.loads(result.content)
        # else:
        #     result_json = json.loads(result)
        return result_json

    def similartiy_result(self, entities):
        # Step 1: Use similarity_candidates for initial filtering
        candidates = self.similarity_candidates(entities)
        
        # Step 2: Fine-grained LLM judgment for each candidate pair
        candidates_result = []
        for ent_pair in candidates:
            # Extract entity objects from entities dictionary
            entity1 = entities.get(ent_pair[0])
            entity2 = entities.get(ent_pair[1])
            
            # Call LLM for judgment
            try:
                result = self.similarity_llm_single(entity1, entity2)
                # Keep if LLM judges as same entity
                if result.get('result', False):
                    candidates_result.append(ent_pair)
            except Exception as e:
                print(f"Error processing entity pair {ent_pair}: {str(e)}")
                continue  # Can log or raise exception as needed
        
        # Step 3: Return final filtered candidate pairs
        return candidates_result

    ## Merge similar items
    def entity_Disambiguation(self, entity_dic, sim_entity_list):
        # Step 1: Build and manage merge relationships using union-find
        parent = {}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_y] = root_x

        # Initialize union-find
        for entity in entity_dic:
            parent[entity] = entity
        for pair in sim_entity_list:
            a, b = pair
            if a in entity_dic and b in entity_dic:
                union(a, b)

        # Step 2: Merge similar entities
        groups = {}
        for entity in entity_dic:
            root = find(entity)
            if root not in groups:
                groups[root] = []
            groups[root].append(entity)

        # Step 3: Process each merge group
        for group in groups.values():
            if len(group) == 1:
                continue

            # Sort by appearance order, keep name/type of first entity
            main_entity = group[0]
            descriptions = set()
            chunkids = set()

            for e in group:
                descriptions.add(entity_dic[e]['description'])
                chunkids.add(entity_dic[e]['chunkid'])
                if e != main_entity:
                    del entity_dic[e]  # Remove merged entities

            # Merge fields
            entity_dic[main_entity]['description'] = ';;;'.join(descriptions)
            entity_dic[main_entity]['chunkid'] = ';;;'.join(chunkids)

        # Step 4: Directly return merged entity dictionary
        return entity_dic

    def get_sentences_for_entity(self,entity_dic, entity_id, id_to_sentence):
        """
        Extract sentences corresponding to chunkid for a specified entity ID.

        Parameters:
            entity_dic (dict): Dictionary containing entity information.
            entity_id (str): Specified entity ID.
            id_to_sentence (dict): Dictionary mapping chunkid to sentences.

        Returns:
            list: List of sentences corresponding to the specified entity's chunkid.
        """
        if entity_id not in entity_dic:
            raise ValueError(f"Entity '{entity_id}' not found in entity_dic.")

        # Get chunkid field for specified entity
        chunkids = entity_dic[entity_id].get('chunkid', '')
        if not chunkids:
            return []  # Return empty list if no chunkid

        # Split chunkid into multiple IDs
        chunkid_list = chunkids.split(';;;')
        chunkid_list = [cid.strip() for cid in chunkid_list if cid.strip()]  # Remove empty strings

        # Find corresponding sentences from id_to_sentence
        sentences = []
        for chunkid in chunkid_list:
            if chunkid in id_to_sentence:
                sentences.append(id_to_sentence[chunkid])
            else:
                print(f"Warning: Chunk ID '{chunkid}' not found in id_to_sentence.")

        return sentences

    def get_retriever_context(self, query, sentences, sentence_to_id,vectors,top_k=5):
        """
        Get the top_k most similar sentences as retriever context for a query.

        :param query: str, user's query text
        :param top_k: int, number of most similar sentences to return, default is 5
        :return: list of tuples, each tuple contains (sentence, similarity, sentence_id)
        """

        # Step 1: Convert query to vector
        query_vector = self.embeddings.embed_query(query)

        # Step 2: Calculate cosine similarity between query vector and sentence vectors
        sentence_vectors = np.array(vectors)
        similarities = cosine_similarity([query_vector], sentence_vectors)[0]

        # Step 3: Select top_k most similar sentences
        top_indices = np.argsort(similarities)[::-1][:top_k]  # Sort by similarity in descending order and take top_k
        retriever_context = []
        for idx in top_indices:
            sentence = sentences[idx]
            similarity = similarities[idx]
            sentence_id = sentence_to_id[sentence]
            retriever_context.append((sentence, similarity, sentence_id))

        return retriever_context
    
    def log_on_retry(self, retry_state: RetryCallState):
        """在每次重试前记录日志"""
        self.logger.warning(
            f"LLM call failed with exception: {retry_state.outcome.exception()}. "
            f"Retrying in {retry_state.next_action.sleep} seconds... "
            f"(Attempt {retry_state.attempt_number})"
        )
    
    @retry(stop=stop_after_attempt(3), 
            wait=wait_fixed(3),
            before_sleep=log_on_retry)
    def call_llm_with_timeout(
            self, 
            chain: Runnable,
            inputs: Dict[str, any]
        ):
        """
        LLM invoke with timeout and auto-retry
        """
        res = chain.invoke(inputs)
        return json.loads(res.content if hasattr(res, "content") else res)

    def get_target_kg_sigle(self, entity_dic, entity_id, id_to_sentence, sentences, sentence_to_id, vectors, output_file):
        chunk_text_list = self.get_sentences_for_entity(entity_dic, entity_id, id_to_sentence)
        query = entity_dic[entity_id].get('name', '')
        context = self.get_retriever_context(query, sentences, sentence_to_id, vectors, top_k=5)
        sentences = [item[0] for item in context]
        unique_sentences = list(set(chunk_text_list + sentences))
        chunk_text = ", ".join(unique_sentences)
        prompt = ChatPromptTemplate.from_template(extract_entiry_centric_kg_en_v2)
        chain = prompt | self.model

        try:
            result_json = self.call_llm_with_timeout(
                chain,
                {
                    "text": chunk_text,
                    "target_entity": entity_dic[entity_id]["name"],
                    "related_kg": "none",
                }
            )
        except Exception as e:
            self.logger.warning(f"[{entity_id}] LLM inoke failure after all retries: {e}")
            result_json = {}

        # result = chain.invoke({"text": chunk_text, "target_entity": entity_dic[entity_id].get('name'), "related_kg": 'none'})
        # # Handle AIMessage response from OpenAI
        # if hasattr(result, 'content'):
        #     result_json = json.loads(result.content)
        # else:
        #     result_json = json.loads(result)

        with open(output_file, 'a') as f:
            combined_data = {
                "chunk_text": chunk_text,
                "entity": entity_dic[entity_id],
                "kg": result_json
            }
            f.write(json.dumps(combined_data, ensure_ascii=False) + '\n')

        return result_json
    
    def get_target_kg_all(self, entity_dic, id_to_sentence,sentences,sentence_to_id,vectors,output_file):
        """
        Process all entities.
        """
        results = {}
        for entity_id in entity_dic:
            self.logger.info(f"Processing entity: {entity_id}: {entity_dic[entity_id]['name']}, len: {len(entity_dic[entity_id]['description'])}")
            if entity_id in entity_dic:
                result = self.get_target_kg_sigle(entity_dic, entity_id, id_to_sentence, sentences, sentence_to_id, vectors,output_file)
                results[entity_id] = result
            else:
                self.logger.warning(f"Entity {entity_id} not found in entity_dic.")
        return results

    def convert_knowledge_graph(self, input_data):
        output = {
            "entities": [],
            "relations": []
        }

        entity_registry = {}

        # First pass: Process original entities
        for entity_key in input_data:
            central_entity = input_data[entity_key]["central_entity"]
            entity_name = central_entity["name"]
            
            if entity_name not in entity_registry:
                entity = {
                    "name": entity_name,
                    "type": central_entity["type"],
                    "description": central_entity.get("description", ""),  # Add description field
                    "attributes": {}
                }
                if "attributes" in central_entity:
                    for attr in normalize_attributes(central_entity["attributes"]): # format check
                        entity["attributes"][attr["key"]] = attr["value"]
                entity_registry[entity_name] = entity

        # Second pass: Process relationships
        for entity_key in input_data:
            central_entity = input_data[entity_key]["central_entity"]
            
            if "relationships" in central_entity:
                
                relationships = central_entity["relationships"]
                relationships_normalized = normalize_relationships(relationships)

                for rel in relationships_normalized:
                    
                    # TODO fix dict format normalized 
                    # rel_normalized = normalize_attributes_dict(rel)
                    rel_normalized = normalize_attributes_dict_origin(rel)
                    if not rel_normalized:
                        continue

                    # Handle target entities that might be lists
                    target_names = rel_normalized["target_name"] if isinstance(rel_normalized["target_name"], list) else [rel_normalized["target_name"]]
                    target_type = rel_normalized["target_type"]
                    
                    for target_name in target_names:
                        if not target_name:
                            continue
                        # Register target entity
                        if target_name not in entity_registry:
                            entity_registry[target_name] = {
                                "name": target_name,
                                "type": target_type,
                                "description": rel_normalized.get("target_description", ""),  # Add description field
                                "attributes": {}
                            }
                        
                        # Add relationship quadruple (including relation_description)
                        relation_description = rel_normalized.get("relation_description", "")
                        output["relations"].append([
                            central_entity["name"],
                            rel_normalized["relation"],
                            target_name,
                            relation_description
                        ])

        output["entities"] = list(entity_registry.values())
        return output
    

