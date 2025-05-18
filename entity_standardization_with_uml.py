import json
import os
import pandas as pd
import numpy as np
import requests
import time
from tqdm import tqdm
import urllib.parse
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("entity_standardization.log"),
        logging.StreamHandler()
    ]
)

class UMLSEntityStandardizer:
    
    def __init__(self, api_key=None, ticket_granting_ticket=None):
        self.api_key = api_key
        self.base_url = "https://uts-ws.nlm.nih.gov/rest"
        self.tgt = ticket_granting_ticket
        self.service_ticket = None
        self.term_cache = {}  # Cache for term lookups
        
        # Maximum number of retries for API calls
        self.max_retries = 5
        # Delay between retries (in seconds)
        self.retry_delay = 2
        
    def authenticate(self):
        if not self.api_key:
            logging.warning("No API key provided. Please obtain a UMLS API key from https://uts.nlm.nih.gov/uts/")
            return None
            
        auth_endpoint = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
        params = {'apikey': self.api_key}
        
        try:
            logging.info("Authenticating with UMLS API...")
            response = requests.post(auth_endpoint, data=params)
            response.raise_for_status()
            
            self.tgt = response.text.split('name="execution"')[1].split('value="')[1].split('"')[0]
            logging.info("Successfully authenticated with UMLS API")
            return self.tgt
            
        except Exception as e:
            logging.error(f"Error authenticating with UMLS API: {e}")
            return None
    
    def get_service_ticket(self):
        if not self.tgt:
            logging.warning("No ticket-granting ticket available. Call authenticate() first.")
            return None
            
        service_ticket_endpoint = f"https://utslogin.nlm.nih.gov/cas/v1/tickets/{self.tgt}"
        params = {'service': 'http://umlsks.nlm.nih.gov'}
        
        try:
            response = requests.post(service_ticket_endpoint, data=params)
            response.raise_for_status()
            self.service_ticket = response.text
            return self.service_ticket
            
        except Exception as e:
            logging.error(f"Error getting service ticket: {e}")
            return None
    
    def search_term(self, term, source=None):
        cache_key = f"{term}_{source}"
        if cache_key in self.term_cache:
            return self.term_cache[cache_key]
            
        if not self.service_ticket:
            self.service_ticket = self.get_service_ticket()
            if not self.service_ticket:
                return []
                
        search_endpoint = f"{self.base_url}/search/current"
        
        params = {
            'string': term,
            'ticket': self.service_ticket,
            'pageSize': 10
        }
        
        if source:
            params['sabs'] = source
            
        retries = 0
        while retries < self.max_retries:
            try:
                response = requests.get(search_endpoint, params=params)
                response.raise_for_status()
                results = response.json()
                
                self.term_cache[cache_key] = results
                return results
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:  # Unauthorized
                    logging.warning("Service ticket expired. Getting a new one...")
                    self.service_ticket = self.get_service_ticket()
                    retries += 1
                elif e.response.status_code == 429:  # Too Many Requests
                    wait_time = self.retry_delay * (2 ** retries)
                    logging.warning(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    logging.error(f"HTTP error during search: {e}")
                    return []
            except Exception as e:
                logging.error(f"Error searching for term '{term}': {e}")
                retries += 1
                time.sleep(self.retry_delay)
                
        logging.error(f"Failed to search for term '{term}' after {self.max_retries} retries")
        return []
    
    def get_concept_details(self, cui):
        if not self.service_ticket:
            self.service_ticket = self.get_service_ticket()
            if not self.service_ticket:
                return {}
                
        concept_endpoint = f"{self.base_url}/content/current/CUI/{cui}"
        
        params = {
            'ticket': self.service_ticket
        }
        
        retries = 0
        while retries < self.max_retries:
            try:
                response = requests.get(concept_endpoint, params=params)
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:  # Unauthorized
                    logging.warning("Service ticket expired. Getting a new one...")
                    self.service_ticket = self.get_service_ticket()
                    retries += 1
                elif e.response.status_code == 429:  # Too Many Requests
                    wait_time = self.retry_delay * (2 ** retries)
                    logging.warning(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    logging.error(f"HTTP error getting concept details: {e}")
                    return {}
            except Exception as e:
                logging.error(f"Error getting details for concept '{cui}': {e}")
                retries += 1
                time.sleep(self.retry_delay)
                
        logging.error(f"Failed to get details for concept '{cui}' after {self.max_retries} retries")
        return {}
    
    def get_atoms_for_concept(self, cui, source=None):
        if not self.service_ticket:
            self.service_ticket = self.get_service_ticket()
            if not self.service_ticket:
                return []
                
        atoms_endpoint = f"{self.base_url}/content/current/CUI/{cui}/atoms"
        
        params = {
            'ticket': self.service_ticket,
            'pageSize': 100
        }
        
        if source:
            params['sabs'] = source
            
        retries = 0
        while retries < self.max_retries:
            try:
                response = requests.get(atoms_endpoint, params=params)
                response.raise_for_status()
                results = response.json()
                return results.get('result', [])
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:  # Unauthorized
                    logging.warning("Service ticket expired. Getting a new one...")
                    self.service_ticket = self.get_service_ticket()
                    retries += 1
                elif e.response.status_code == 429:  # Too Many Requests
                    wait_time = self.retry_delay * (2 ** retries)
                    logging.warning(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    logging.error(f"HTTP error getting atoms: {e}")
                    return []
            except Exception as e:
                logging.error(f"Error getting atoms for concept '{cui}': {e}")
                retries += 1
                time.sleep(self.retry_delay)
                
        logging.error(f"Failed to get atoms for concept '{cui}' after {self.max_retries} retries")
        return []
    
    def map_to_rxnorm(self, drug_name):
        if not drug_name or not drug_name.strip():
            return {
                'original': drug_name,
                'standardized': None,
                'cui': None,
                'code': None,
                'source': 'RXNORM',
                'status': 'empty'
            }
            
        results = self.search_term(drug_name, 'RXNORM')
        
        if not results or 'result' not in results or not results['result']:
            return {
                'original': drug_name,
                'standardized': None,
                'cui': None,
                'code': None,
                'source': 'RXNORM',
                'status': 'not_found'
            }
            
        best_match = results['result']['results'][0]
        cui = best_match['ui']
        
        atoms = self.get_atoms_for_concept(cui, 'RXNORM')
        
        preferred_term = None
        rxnorm_code = None
        
        for atom in atoms:
            if atom.get('termType') == 'PT':  # Preferred Term
                preferred_term = atom.get('name')
                rxnorm_code = atom.get('code')
                break
                
        if not preferred_term:
            preferred_term = best_match.get('name')
            
        return {
            'original': drug_name,
            'standardized': preferred_term,
            'cui': cui,
            'code': rxnorm_code,
            'source': 'RXNORM',
            'status': 'found'
        }
    
    def map_to_snomed(self, term):
        if not term or not term.strip():
            return {
                'original': term,
                'standardized': None,
                'cui': None,
                'code': None,
                'source': 'SNOMEDCT_US',
                'status': 'empty'
            }
            
        results = self.search_term(term, 'SNOMEDCT_US')
        
        if not results or 'result' not in results or not results['result']:
            return {
                'original': term,
                'standardized': None,
                'cui': None,
                'code': None,
                'source': 'SNOMEDCT_US',
                'status': 'not_found'
            }
            
        best_match = results['result']['results'][0]
        cui = best_match['ui']
        
        atoms = self.get_atoms_for_concept(cui, 'SNOMEDCT_US')
        preferred_term = None
        snomed_code = None
        
        for atom in atoms:
            if atom.get('termType') == 'PT':  # Preferred Term
                preferred_term = atom.get('name')
                snomed_code = atom.get('code')
                break
                
        if not preferred_term:
            preferred_term = best_match.get('name')
            
        return {
            'original': term,
            'standardized': preferred_term,
            'cui': cui,
            'code': snomed_code,
            'source': 'SNOMEDCT_US',
            'status': 'found'
        }
    
    def standardize_entities(self, extraction_results, limit=None):
        if limit:
            extraction_results = extraction_results.head(limit)
            
        standardized_results = []
        
        for _, row in tqdm(extraction_results.iterrows(), total=len(extraction_results), desc="Standardizing entities"):
            post_id = row['id']
            entities = row['entities']
            
            standardized_drugs = []

            if isinstance(entities, str):
                entities = json.loads(entities)

            for drug in entities.get('drugs', []):
                drug_name = drug.get('name', drug.get('span', ''))
                standardized_drug = self.map_to_rxnorm(drug_name)
                
                standardized_drug['span'] = drug.get('span', '')
                standardized_drugs.append(standardized_drug)
                
                time.sleep(0.1)
            
            standardized_ades = []
            for ade in entities.get('adverse_events', []):
                ade_description = ade.get('description', ade.get('span', ''))
                standardized_ade = self.map_to_snomed(ade_description)
                
                standardized_ade['span'] = ade.get('span', '')
                standardized_ades.append(standardized_ade)
                
                time.sleep(0.1)
            
            standardized_symptoms = []
            for symptom in entities.get('symptoms_diseases', []):
                symptom_name = symptom.get('name', symptom.get('span', ''))
                standardized_symptom = self.map_to_snomed(symptom_name)
                
                standardized_symptom['span'] = symptom.get('span', '')
                standardized_symptoms.append(standardized_symptom)
                time.sleep(0.1)
            
            standardized_entities = {
                'drugs': standardized_drugs,
                'adverse_events': standardized_ades,
                'symptoms_diseases': standardized_symptoms
            }
            
            standardized_results.append({
                'id': post_id,
                'entities': entities,  # Original extracted entities
                'standardized_entities': standardized_entities  # Standardized entities
            })
            
        return pd.DataFrame(standardized_results)
    
    def standardize_from_file(self, extraction_results_path, output_path=None, limit=None):
        logging.info(f"Loading extraction results from {extraction_results_path}...")
        extraction_results = pd.read_pickle(extraction_results_path)
        
        logging.info("Starting entity standardization...")
        standardized_df = self.standardize_entities(extraction_results, limit)

        if output_path:
            logging.info(f"Saving standardized results to {output_path}...")
            standardized_df.to_pickle(output_path)
            
            # Also save a sample to JSON for easy inspection
            sample_path = output_path.replace('.pkl', '_sample.json')
            sample_df = standardized_df.head(10)
            
            with open(sample_path, 'w', encoding='utf-8') as f:
                json.dump(sample_df.to_dict('records'), f, indent=2)
            
            logging.info(f"Saved sample to {sample_path}")
        
        return standardized_df
    
    def generate_standardization_report(self, standardized_df, output_path=None):
        logging.info("Generating standardization report...")
        
        total_drugs = 0
        standardized_drugs = 0
        total_ades = 0
        standardized_ades = 0
        total_symptoms = 0
        standardized_symptoms = 0
        
        for _, row in standardized_df.iterrows():
            standardized_entities = row['standardized_entities']
            
            drugs = standardized_entities.get('drugs', [])
            total_drugs += len(drugs)
            standardized_drugs += sum(1 for drug in drugs if drug.get('status') == 'found')
            
            ades = standardized_entities.get('adverse_events', [])
            total_ades += len(ades)
            standardized_ades += sum(1 for ade in ades if ade.get('status') == 'found')
            
            symptoms = standardized_entities.get('symptoms_diseases', [])
            total_symptoms += len(symptoms)
            standardized_symptoms += sum(1 for symptom in symptoms if symptom.get('status') == 'found')
        
        drug_rate = standardized_drugs / total_drugs if total_drugs > 0 else 0
        ade_rate = standardized_ades / total_ades if total_ades > 0 else 0
        symptom_rate = standardized_symptoms / total_symptoms if total_symptoms > 0 else 0
        overall_rate = (standardized_drugs + standardized_ades + standardized_symptoms) / (total_drugs + total_ades + total_symptoms) if (total_drugs + total_ades + total_symptoms) > 0 else 0
        
        report = {
            'total_posts': len(standardized_df),
            'drugs': {
                'total': total_drugs,
                'standardized': standardized_drugs,
                'rate': drug_rate
            },
            'adverse_events': {
                'total': total_ades,
                'standardized': standardized_ades,
                'rate': ade_rate
            },
            'symptoms_diseases': {
                'total': total_symptoms,
                'standardized': standardized_symptoms,
                'rate': symptom_rate
            },
            'overall': {
                'total': total_drugs + total_ades + total_symptoms,
                'standardized': standardized_drugs + standardized_ades + standardized_symptoms,
                'rate': overall_rate
            }
        }
        
        logging.info("Standardization Report:")
        logging.info(f"Total posts: {report['total_posts']}")
        logging.info(f"Drugs: {report['drugs']['standardized']}/{report['drugs']['total']} standardized ({report['drugs']['rate']:.2%})")
        logging.info(f"Adverse events: {report['adverse_events']['standardized']}/{report['adverse_events']['total']} standardized ({report['adverse_events']['rate']:.2%})")
        logging.info(f"Symptoms/diseases: {report['symptoms_diseases']['standardized']}/{report['symptoms_diseases']['total']} standardized ({report['symptoms_diseases']['rate']:.2%})")
        logging.info(f"Overall: {report['overall']['standardized']}/{report['overall']['total']} standardized ({report['overall']['rate']:.2%})")

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            
            logging.info(f"Saved standardization report to {output_path}")
        
        return report

if __name__ == "__main__":
    extraction_results_path = "./processed_data/entity_extraction_results.pkl"
    output_path = "./processed_data/standardized_entities.pkl"
    report_path = "./processed_data/standardization_report.json"
    
    standardizer = UMLSEntityStandardizer(api_key="your_umls_api_key")
    
    standardizer.authenticate()
    
    standardized_df = standardizer.standardize_from_file(
        extraction_results_path=extraction_results_path,
        output_path=output_path,
        limit=5
    )
    
    standardizer.generate_standardization_report(standardized_df, report_path)