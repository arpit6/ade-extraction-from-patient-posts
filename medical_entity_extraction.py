import json
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import os

class MedicalEntityExtractor:

    def __init__(self, model_name="phamhai/Llama-3.2-3B-Instruct-Frog", max_length=512):       
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.llm_chain = None
        self.initialize_model()

    def initialize_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained("phamhai/Llama-3.2-3B-Instruct-Frog")
        self.model = AutoModelForCausalLM.from_pretrained(
            "phamhai/Llama-3.2-3B-Instruct-Frog",
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def clean_json_output(self, output):
        pattern = r'\[Assistant\]\s*({.*?})\s*$'

        default_json = {
            "drugs": [{"name": "", "span": ""}],
            "adverse_events": [{"description": "", "span": ""}],
            "symptoms_diseases": [{"name": "", "span": ""}]
        }


        match = re.search(pattern, output, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                json_obj = json.loads(json_str)
                return json.dumps(json_obj, indent=2)
            except json.JSONDecodeError:
                return json.dumps(default_json, indent=2)
        else:
            return json.dumps(default_json, indent=2)


    def extract_user_query(self, query):
        prompt = prompt = f"""
               Task: Extract medical entities from the patient forum post below.

            Post: {query}

            Instructions:
            - Use only the **exact words** found in the post.
            - Do **not** infer, assume, or add anything not explicitly written in the post.
            - Output a single JSON object with the following keys:
            - "drugs": list of drug names (exactly as written in the post)
            - "adverse_events": list of adverse effects (verbatim)
            - "symptoms_diseases": list of symptoms or diseases (verbatim)
            - If no items are found in a category, return an empty list for that key.
            - Do not add commentary, explanation, or multiple JSON blocks.

            Output Format:
            ```json
                Extract the following entities:
                1. Drugs/Medications: Any medicine, drug, or pharmaceutical product mentioned
                2. Adverse Drug Events (ADEs): Any negative effect or adverse reaction related to medication
                3. Symptoms/Diseases: Any medical condition, symptom, or disease mentioned
                
                Return the results in the following JSON format:
                {{
                    "drugs": [{{
                        "name": "drug name",
                        "span": "exact text span from post"
                    }}],
                    "adverse_events": [{{
                        "description": "adverse event description",
                        "span": "exact text span from post"
                    }}],
                    "symptoms_diseases": [{{
                        "name": "symptom or disease name",
                        "span": "exact text span from post"
                    }}]
                }}
            """
        messages = [
            {"role": "assistant", "content": "You are an assistant with knowledge of medical data extraction"},
            {"role": "user", "content": prompt}
        ]
        return self.send_prompt(messages)


    def send_prompt(self, messages, temperature=0.2):
        formatted = self.format_messages(messages)

        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)


    def format_messages(self, messages):
        prompt = ""
        for msg in messages:
            role = msg['role'].capitalize()
            content = msg['content']
            prompt += f"[{role}]\n{content}\n"
        return prompt + "[Assistant]\n"



    def extract_entities(self, text):
        raw_output = self.extract_user_query(text)
        entities = self.clean_json_output(raw_output)
        return entities
    
    def batch_extract_entities(self, posts, batch_size=10):

        results = []
        
        if isinstance(posts, pd.DataFrame):
            text_key = 'normalized_text' if 'normalized_text' in posts.columns else 'text'
            post_list = [{'id': row['id'], 'text': row[text_key]} for _, row in posts.iterrows()]
        else:
            post_list = posts

        total_batches = (len(post_list) + batch_size - 1) // batch_size
        
        for i in tqdm(range(total_batches), desc="Extracting medical entities"):
            batch = post_list[i * batch_size: (i + 1) * batch_size]
            
            for post in batch:
                post_id = post.get('id', f"post_{len(results)}")
                post_text = post.get('text', '')
                
                entities = self.extract_entities(post_text)

                result = {
                    'id': post_id,
                    'entities': entities
                }
                
                results.append(result)

        return results
    
    def extract_from_processed_data(self, processed_data_path, output_path=None, batch_size=10):

        print(f"Loading processed data from {processed_data_path}...")
        processed_df = pd.read_pickle(processed_data_path)
        
        print(f"Extracting entities from {len(processed_df)} posts...")
        extraction_results = self.batch_extract_entities(processed_df, batch_size)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(extraction_results)
        
        # Save results if output path is provided
        if output_path:
            print(f"Saving extraction results to {output_path}...")
            results_df.to_pickle(output_path)
        
        return results_df
    


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_path = os.path.join(base_dir, "processed_data/cadec_processed.pkl")
    extraction_output_path = os.path.join(base_dir, "processed_data/entity_extraction_results.pkl")

    if not os.path.exists(processed_data_path):
        print(f"Error: Processed data file not found at {processed_data_path}")
        print("Please run the preprocessing step first.")
        return
    
    # Initialize the entity extractor
    print("Initializing Medical Entity Extractor...")
    entity_extractor = MedicalEntityExtractor(
        model_name="phamhai/Llama-3.2-3B-Instruct-Frog"
    )
    
    # Extract entities
    print(f"Extracting entities from {processed_data_path}...")
    results_df = entity_extractor.extract_from_processed_data(
        processed_data_path=processed_data_path,
        output_path=extraction_output_path,
        batch_size=5
    )
    
    print("\nSample extraction results:")
    for _, row in results_df.head(3).iterrows():
        print(f"Post ID: {row['id']}")
        
        if isinstance(row['entities'], str):
            row['entities'] = json.loads(row['entities'])
        print("Drugs:")
        for drug in row['entities'].get('drugs', []):
            print(f"  - {drug['name']} (span: '{drug['span']}')")
        
        print("Adverse Events:")
        for ade in row['entities'].get('adverse_events', []):
            print(f"  - {ade['description']} (span: '{ade['span']}')")
        
        print("Symptoms/Diseases:")
        for symptom in row['entities'].get('symptoms_diseases', []):
            print(f"  - {symptom['name']} (span: '{symptom['span']}')")
        
        print("-" * 50)
    
    # Print statistics
    print("\nExtraction statistics:")
    total_drugs = sum(len(row['entities'].get('drugs', [])) for _, row in results_df.iterrows())
    total_ades = sum(len(row['entities'].get('adverse_events', [])) for _, row in results_df.iterrows())
    total_symptoms = sum(len(row['entities'].get('symptoms_diseases', [])) for _, row in results_df.iterrows())
    
    print(f"Total posts processed: {len(results_df)}")
    print(f"Total drugs extracted: {total_drugs} (avg: {total_drugs/len(results_df):.2f} per post)")
    print(f"Total adverse events extracted: {total_ades} (avg: {total_ades/len(results_df):.2f} per post)")
    print(f"Total symptoms/diseases extracted: {total_symptoms} (avg: {total_symptoms/len(results_df):.2f} per post)")
    
    # Export a sample of results to JSON for easy inspection
    sample_results = results_df.head(10).to_dict('records')

    sample_output_path = os.path.join(base_dir, "processed_data/sample_extraction_results.json")
    
    print(f"\nExporting sample results to {sample_output_path}...")
    with open(sample_output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_results, f, indent=2)
    
    print(f"\nEntity extraction complete. Full results saved to {extraction_output_path}")

if __name__ == "__main__":
    main()