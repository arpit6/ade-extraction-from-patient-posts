
import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline, AutoTokenizer

class CADECPreprocessor:

    def __init__(self, cadec_dir):
        self.cadec_dir = cadec_dir
        self.tokenizer = None
        self.model = None
        self.drug_name_map = {}
        self.initialize_model()

    def initialize_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained("phamhai/Llama-3.2-3B-Instruct-Frog")
        self.model = AutoModelForCausalLM.from_pretrained(
            "phamhai/Llama-3.2-3B-Instruct-Frog",
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def expanded_text(self, text):
        prompt =f"""
            You are a medical assistant.

            Task: Expand medical abbreviations in the text below. Replace abbreviations inline with their full forms and preserve all other text exactly.

            Text:
            {text}

            Output:
            (Return only the expanded text. Do not include any explanations, bullet points, or notes.)
        """
        
        messages = [
            {"role": "assistant", "content": "You are a medical assistant that expands abbreviations inline and outputs only the updated text."},
            {"role": "user", "content": prompt}
        ]
        response_text =  self.send_prompt(messages)

        matches = re.findall(r"\[Assistant\]\n(.*?)(?=\n\[|\Z)", response_text, re.DOTALL)

        if matches:
            assistant_response = matches[-1].strip()
            return re.sub(r'\s+', ' ', assistant_response).strip()
        else:
            return text


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

        
    def load_cadec_data(self):
        print("Loading CADEC forum posts...")
        posts_dir = os.path.join(self.cadec_dir, "text")
        annotations_dir = os.path.join(self.cadec_dir, "original")
        
        if not os.path.exists(posts_dir):
            raise FileNotFoundError(f"CADEC posts directory not found: {posts_dir}")
        
        posts = []
        
        for filename in tqdm(os.listdir(posts_dir)):
            if filename.endswith(".txt"):
                post_id = filename[:-4]
                post_path = os.path.join(posts_dir, filename)
                
                with open(post_path, 'r', encoding='utf-8') as f:
                    post_text = f.read().strip()

                ade_path = os.path.join(annotations_dir, f"{post_id}.ann")
                annotations = []
                
                if os.path.exists(ade_path):
                    with open(ade_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) >= 3:
                                ann_id, ann_info, ann_text = parts
                                ann_type = ann_info.split(' ')[0]
                                annotations.append({
                                    'id': ann_id,
                                    'type': ann_type,
                                    'text': ann_text
                                })
                
                posts.append({
                    'id': post_id,
                    'text': post_text,
                    'annotations': annotations
                })
        
        print(f"Loaded {len(posts)} forum posts.")
        return posts
    
    def expand_abbreviations(self, posts):
        print("Expanding medical abbreviations...")
        for post in tqdm(posts):
            post['expanded_text'] = self.expanded_text(post['text'])
        return posts
    
    def normalize_drug_names(self, posts):
        print("Normalizing drug names...")
        self.drug_name_map = {
            "tylenol": "acetaminophen",
            "advil": "ibuprofen",
            "motrin": "ibuprofen",
            "aspirin": "acetylsalicylic acid",
            "lipitor": "atorvastatin",
            "zocor": "simvastatin",
            "crestor": "rosuvastatin",
            "nexium": "esomeprazole",
            "prilosec": "omeprazole",
            "prozac": "fluoxetine",
            "zoloft": "sertraline",
            "lexapro": "escitalopram",
            "cymbalta": "duloxetine",
            "xanax": "alprazolam",
            "valium": "diazepam",
            "ativan": "lorazepam",
            "ambien": "zolpidem",
            "lunesta": "eszopiclone",
            "lyrica": "pregabalin",
            "neurontin": "gabapentin",
            "dilantin": "phenytoin",
            "coumadin": "warfarin",
            "paxil": "paroxetine",
            "levaquin": "levofloxacin",
            "cipro": "ciprofloxacin",
            "augmentin": "amoxicillin/clavulanate",
            "zithromax": "azithromycin",
            "z-pak": "azithromycin",
            "diflucan": "fluconazole",
            "flagyl": "metronidazole",
            "amoxil": "amoxicillin",
            "keflex": "cephalexin",
            "bactrim": "trimethoprim/sulfamethoxazole"
        }
        
        def normalize_drug(text):
            lower_text = text.lower()
            for brand, generic in self.drug_name_map.items():
                pattern = r'\b' + re.escape(brand) + r'\b'
                lower_text = re.sub(pattern, generic, lower_text)
            return lower_text

        for post in tqdm(posts):
            post['normalized_text'] = normalize_drug(post['expanded_text'])
            
            for ann in post.get('annotations', []):
                if ann['type'] == 'Drug':
                    ann['normalized_text'] = normalize_drug(ann['text'])
                else:
                    ann['normalized_text'] = ann['text']
        
        return posts
    
    def tokenize_text(self, posts):
        print("Tokenizing text...")
        
        for post in tqdm(posts):
            tokenized = self.tokenizer(post['normalized_text'], return_tensors="pt", truncation=True, max_length=512)
            post['tokens'] = tokenized

            post['token_count'] = len(tokenized['input_ids'][0])
        
        return posts
    
    def process(self):
        posts = self.load_cadec_data()

        posts = self.expand_abbreviations(posts)

        posts = self.normalize_drug_names(posts)
        posts = self.tokenize_text(posts)
        

        # Convert to DataFrame for easier handling
        df = pd.DataFrame([{
            'id': post['id'],
            'original_text': post['text'],
            'expanded_text': post['expanded_text'],
            'normalized_text': post['normalized_text'],
            'token_count': post.get('token_count', 0),
            'annotations': post.get('annotations', [])
        } for post in posts])
        
        print(f"Preprocessing complete. Processed {len(posts)} posts.")
        return posts, df
    
    def save_processed_data(self, df, output_path):
        print(f"Saving processed data to {output_path}...")
        df.to_pickle(output_path)
        print("Data saved successfully.")

def main():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    cadec_dir = os.path.join(base_dir, "cadec")
    output_path = os.path.join(base_dir, "processed_data/cadec_processed.pkl")

    print("Initializing CADEC preprocessor...")
    preprocessor = CADECPreprocessor(
        cadec_dir=cadec_dir
    )
    
    print("Starting preprocessing pipeline...")
    posts, df = preprocessor.process()
    
    preprocessor.save_processed_data(df, output_path)

    print("\nSample of processed data:")
    sample_df = df.head(1)
    for _, row in sample_df.iterrows():
        print(f"Post ID: {row['id']}")
        print(f"Original text (excerpt): {row['original_text'][:100]}...")
        print(f"Expanded text (excerpt): {row['expanded_text'][:100]}...")
        print(f"Normalized text (excerpt): {row['normalized_text'][:100]}...")
        print(f"Token count: {row['token_count']}")
        print(f"Annotations: {row['annotations'][:3]}")  # Show first 3 annotations
        print("-" * 50)
    
    print(f"\nPreprocessing complete. Processed data saved to {output_path}")

    print("\nDataset statistics:")
    print(f"Total number of posts: {len(df)}")
    print(f"Average token count: {df['token_count'].mean():.1f}")
    annotation_types = {}
    for annotations in df['annotations']:
        for ann in annotations:
            ann_type = ann['type']
            annotation_types[ann_type] = annotation_types.get(ann_type, 0) + 1
    
    print("\nAnnotation counts by type:")
    for ann_type, count in annotation_types.items():
        print(f"- {ann_type}: {count}")

if __name__ == "__main__":
    main()