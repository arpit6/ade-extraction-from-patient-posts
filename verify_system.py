import json
import os
import pandas as pd
import numpy as np
from jsonschema import validate, ValidationError
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from tqdm import tqdm

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("entity_verification.log"),
        logging.StreamHandler()
    ]
)

class EntityVerificationSystem:
    """
    A verification system for validating extracted and standardized medical entities.
    Implements three types of checks:
    1. Format Verification: Ensures valid JSON schema
    2. Completeness Check: Compares with ground truth annotations
    3. Semantic Similarity Check: Validates entity correctness using cosine similarity
    """
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the verification system with the sentence transformer model.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        logging.info(f"Initializing EntityVerificationSystem with model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            logging.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading sentence transformer model: {e}")
            self.model = None
        
        # Entity schema definitions for validation
        self.entity_schema = {
            "type": "object",
            "properties": {
                "drugs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "span": {"type": "string"},
                            "original": {"type": "string"},
                            "standardized": {"type": ["string", "null"]},
                            "cui": {"type": ["string", "null"]},
                            "code": {"type": ["string", "null"]},
                            "source": {"type": "string"},
                            "status": {"type": "string"}
                        },
                        "required": ["original", "status"]
                    }
                },
                "adverse_events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "span": {"type": "string"},
                            "original": {"type": "string"},
                            "standardized": {"type": ["string", "null"]},
                            "cui": {"type": ["string", "null"]},
                            "code": {"type": ["string", "null"]},
                            "source": {"type": "string"},
                            "status": {"type": "string"}
                        },
                        "required": ["original", "status"]
                    }
                },
                "symptoms_diseases": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "span": {"type": "string"},
                            "original": {"type": "string"},
                            "standardized": {"type": ["string", "null"]},
                            "cui": {"type": ["string", "null"]},
                            "code": {"type": ["string", "null"]},
                            "source": {"type": "string"},
                            "status": {"type": "string"}
                        },
                        "required": ["original", "status"]
                    }
                }
            },
            "required": ["drugs", "adverse_events", "symptoms_diseases"]
        }
    
    def format_verification(self, entity_data):
        """
        Verify that entity data conforms to the expected JSON schema.
        
        Args:
            entity_data (dict): Entity data to verify
            
        Returns:
            dict: Verification result with status and errors
        """
        if not isinstance(entity_data, dict):
            return {
                "status": "failed",
                "errors": ["Entity data must be a dictionary"]
            }
            
        try:
            validate(instance=entity_data, schema=self.entity_schema)
            return {
                "status": "passed",
                "errors": []
            }
        except ValidationError as e:
            return {
                "status": "failed",
                "errors": [str(e)]
            }
    
    def calculate_semantic_similarity(self, text1, text2):
        """
        Calculate the semantic similarity between two text strings.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Cosine similarity score between the two texts
        """
        if not self.model:
            logging.error("Sentence transformer model is not loaded")
            return 0.0
            
        if not text1 or not text2:
            return 0.0
            
        try:
            # Encode the texts
            embedding1 = self.model.encode([text1])[0]
            embedding2 = self.model.encode([text2])[0]
            
            # Calculate cosine similarity
            sim_score = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(sim_score)
        except Exception as e:
            logging.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def completeness_check(self, extracted_entities, ground_truth, similarity_threshold=0.7):
        """
        Compare extracted entities with ground truth annotations.
        
        Args:
            extracted_entities (dict): Extracted entities
            ground_truth (dict): Ground truth annotations
            similarity_threshold (float): Threshold for semantic similarity
            
        Returns:
            dict: Completeness check results
        """
        if not isinstance(extracted_entities, dict) or not isinstance(ground_truth, dict):
            return {
                "status": "failed",
                "errors": ["Invalid input format for completeness check"],
                "metrics": {}
            }
            
        results = {
            "drugs": {"found": 0, "missed": 0, "extra": 0, "matches": []},
            "adverse_events": {"found": 0, "missed": 0, "extra": 0, "matches": []},
            "symptoms_diseases": {"found": 0, "missed": 0, "extra": 0, "matches": []}
        }
        
        # Create combined ground truth for ADEs and symptoms (they might be in different categories)
        combined_gt_symptoms_ades = ground_truth.get("adverse_events", []) + ground_truth.get("symptoms_diseases", [])
        combined_extracted_symptoms_ades = extracted_entities.get("adverse_events", []) + extracted_entities.get("symptoms_diseases", [])
        
        # Check drugs
        gt_drugs = [d.get("name", "").lower() for d in ground_truth.get("drugs", [])]
        extracted_drugs = [d.get("original", "").lower() for d in extracted_entities.get("drugs", [])]
        
        # Find matches for drugs
        matches = []
        for gt_drug in gt_drugs:
            best_match = None
            best_score = 0
            
            for extracted_drug in extracted_drugs:
                if not extracted_drug:
                    continue
                    
                score = self.calculate_semantic_similarity(gt_drug, extracted_drug)
                if score > similarity_threshold and score > best_score:
                    best_match = extracted_drug
                    best_score = score
            
            if best_match:
                matches.append((gt_drug, best_match, best_score))
                results["drugs"]["found"] += 1
            else:
                results["drugs"]["missed"] += 1
                
        results["drugs"]["matches"] = matches
        results["drugs"]["extra"] = max(0, len(extracted_drugs) - len(matches))
        
        # Check adverse events and symptoms (combined)
        gt_symptoms_ades = [item.get("description", item.get("name", "")).lower() 
                           for item in combined_gt_symptoms_ades]
        extracted_symptoms_ades = [item.get("original", "").lower() 
                                  for item in combined_extracted_symptoms_ades]
        
        # Find matches for symptoms/ADEs
        matches = []
        for gt_item in gt_symptoms_ades:
            best_match = None
            best_score = 0
            
            for extracted_item in extracted_symptoms_ades:
                if not extracted_item:
                    continue
                    
                score = self.calculate_semantic_similarity(gt_item, extracted_item)
                if score > similarity_threshold and score > best_score:
                    best_match = extracted_item
                    best_score = score
            
            if best_match:
                matches.append((gt_item, best_match, best_score))
                # Increment either adverse_events or symptoms_diseases based on best match category
                results["adverse_events"]["found"] += 1  # Simplified: just count as ADE for now
            else:
                results["adverse_events"]["missed"] += 1
                
        results["adverse_events"]["matches"] = matches
        results["adverse_events"]["extra"] = max(0, len(extracted_symptoms_ades) - len(matches))
        
        # Calculate overall metrics
        total_gt = len(gt_drugs) + len(gt_symptoms_ades)
        total_found = results["drugs"]["found"] + results["adverse_events"]["found"]
        total_extracted = len(extracted_drugs) + len(extracted_symptoms_ades)
        
        if total_gt > 0:
            recall = total_found / total_gt
        else:
            recall = 0.0
            
        if total_extracted > 0:
            precision = total_found / total_extracted
        else:
            precision = 0.0
            
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
            
        results["metrics"] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
        
        status = "passed" if f1_score >= 0.7 else "partial" if f1_score >= 0.4 else "failed"
        
        return {
            "status": status,
            "errors": [],
            "results": results
        }
    
    def semantic_similarity_check(self, standardized_entities, similarity_threshold=0.7):
        """
        Validate entity correctness using semantic similarity.
        Checks if standardized terms are semantically similar to original terms.
        
        Args:
            standardized_entities (dict): Standardized entities
            similarity_threshold (float): Threshold for semantic similarity
            
        Returns:
            dict: Semantic similarity check results
        """
        if not isinstance(standardized_entities, dict):
            return {
                "status": "failed",
                "errors": ["Invalid input format for semantic similarity check"],
                "results": {}
            }
            
        results = {
            "drugs": {"total": 0, "valid": 0, "scores": []},
            "adverse_events": {"total": 0, "valid": 0, "scores": []},
            "symptoms_diseases": {"total": 0, "valid": 0, "scores": []}
        }
        
        # Check drugs
        for drug in standardized_entities.get("drugs", []):
            original = drug.get("original", "")
            standardized = drug.get("standardized")
            
            if original and standardized:
                results["drugs"]["total"] += 1
                similarity = self.calculate_semantic_similarity(original, standardized)
                results["drugs"]["scores"].append({
                    "original": original,
                    "standardized": standardized,
                    "similarity": similarity
                })
                
                if similarity >= similarity_threshold:
                    results["drugs"]["valid"] += 1
        
        # Check adverse events
        for ade in standardized_entities.get("adverse_events", []):
            original = ade.get("original", "")
            standardized = ade.get("standardized")
            
            if original and standardized:
                results["adverse_events"]["total"] += 1
                similarity = self.calculate_semantic_similarity(original, standardized)
                results["adverse_events"]["scores"].append({
                    "original": original,
                    "standardized": standardized,
                    "similarity": similarity
                })
                
                if similarity >= similarity_threshold:
                    results["adverse_events"]["valid"] += 1
        
        # Check symptoms/diseases
        for symptom in standardized_entities.get("symptoms_diseases", []):
            original = symptom.get("original", "")
            standardized = symptom.get("standardized")
            
            if original and standardized:
                results["symptoms_diseases"]["total"] += 1
                similarity = self.calculate_semantic_similarity(original, standardized)
                results["symptoms_diseases"]["scores"].append({
                    "original": original,
                    "standardized": standardized,
                    "similarity": similarity
                })
                
                if similarity >= similarity_threshold:
                    results["symptoms_diseases"]["valid"] += 1
        
        # Calculate overall metrics
        total_entities = (results["drugs"]["total"] + 
                         results["adverse_events"]["total"] + 
                         results["symptoms_diseases"]["total"])
        
        valid_entities = (results["drugs"]["valid"] + 
                         results["adverse_events"]["valid"] + 
                         results["symptoms_diseases"]["valid"])
        
        if total_entities > 0:
            validity_rate = valid_entities / total_entities
        else:
            validity_rate = 0.0
            
        results["metrics"] = {
            "total_entities": total_entities,
            "valid_entities": valid_entities,
            "validity_rate": validity_rate
        }
        
        status = "passed" if validity_rate >= 0.8 else "partial" if validity_rate >= 0.5 else "failed"
        
        return {
            "status": status,
            "errors": [],
            "results": results
        }
    
    def verify_single_post(self, standardized_entities, ground_truth=None):

        # print('---------------')
        # print(standardized_entities)
        # print(ground_truth)

        """
        Run all verification checks on a single post.
        
        Args:
            standardized_entities (dict): Standardized entities for the post
            ground_truth (dict, optional): Ground truth annotations for the post
            
        Returns:
            dict: Combined verification results
        """
        # Run format verification
        format_check = self.format_verification(standardized_entities)
        print(format_check)
        
        # Run semantic similarity check
        semantic_check = self.semantic_similarity_check(standardized_entities)


        # print('---------------')

        # print(format_check)
        # print(semantic_check)
        
        # Run completeness check if ground truth is provided
        completeness_check = None
        if ground_truth:
            completeness_check = self.completeness_check(standardized_entities, ground_truth)
        
        # Combine results
        verification_results = {
            "format_verification": format_check,
            "semantic_similarity": semantic_check
        }
        
        if completeness_check:
            verification_results["completeness_check"] = completeness_check
        
        # Determine overall status
        statuses = [
            format_check["status"],
            semantic_check["status"]
        ]
        
        if completeness_check:
            statuses.append(completeness_check["status"])
        
        if "failed" in statuses:
            overall_status = "failed"
        elif "partial" in statuses:
            overall_status = "partial"
        else:
            overall_status = "passed"
            
        verification_results["overall_status"] = overall_status
        
        return verification_results
    
    def verify_dataset(self, standardized_df, ground_truth_df=None, output_path=None):
        """
        Run verification on the entire dataset.
        
        Args:
            standardized_df (DataFrame): DataFrame with standardized entities
            ground_truth_df (DataFrame, optional): DataFrame with ground truth annotations
            output_path (str, optional): Path to save verification results
            
        Returns:
            dict: Overall verification results
        """
        logging.info("Starting verification of the dataset...")
        verification_results = []
        
        # Create a dictionary for faster ground truth lookup
        ground_truth_dict = {}

        if ground_truth_df is not None:
            for _, row in ground_truth_df.iterrows():
                post_id = row.get('id')
                if post_id:
                    ground_truth_dict[post_id] = row.to_dict()
        
        # Process each post
        for _, row in tqdm(standardized_df.iterrows(), 
                          total=len(standardized_df), 
                          desc="Verifying entities"):
            post_id = row.get('id')
            standardized_entities = row.get('standardized_entities', {})
            
            # Get ground truth for this post if available
            ground_truth = ground_truth_dict.get(post_id)
            
            if(ground_truth):
                ground_truth = ground_truth['original_text']
            # Run verification checks
            result = self.verify_single_post(standardized_entities, ground_truth)
            
            # Add post ID
            result['id'] = post_id
            verification_results.append(result)
        
        # Create results DataFrame
        results_df = pd.DataFrame(verification_results)
        
        # Calculate overall stats
        total_posts = len(results_df)
        passed_posts = sum(1 for status in results_df['overall_status'] if status == 'passed')
        partial_posts = sum(1 for status in results_df['overall_status'] if status == 'partial')
        failed_posts = sum(1 for status in results_df['overall_status'] if status == 'failed')
        
        overall_results = {
            'total_posts': total_posts,
            'passed_posts': passed_posts,
            'partial_posts': partial_posts,
            'failed_posts': failed_posts,
            'pass_rate': passed_posts / total_posts if total_posts > 0 else 0,
            'failure_rate': failed_posts / total_posts if total_posts > 0 else 0
        }
        
        # Save results if output path is provided
        if output_path:
            logging.info(f"Saving verification results to {output_path}...")
            results_df.to_pickle(output_path)
            
            # Also save a sample to JSON for easy inspection
            sample_path = output_path.replace('.pkl', '_sample.json')
            sample_df = results_df.head(10)
            
            with open(sample_path, 'w', encoding='utf-8') as f:
                json.dump(sample_df.to_dict('records'), f, indent=2)
            
            # Save overall results
            overall_path = output_path.replace('.pkl', '_summary.json')
            with open(overall_path, 'w', encoding='utf-8') as f:
                json.dump(overall_results, f, indent=2)
            
            logging.info(f"Saved results summary to {overall_path}")
        
        logging.info(f"Verification complete. Overall pass rate: {overall_results['pass_rate']:.2%}")
        return overall_results


# Example usage
if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.abspath(__file__))

    standardized_path = os.path.join(base_dir, "processed_data/standardized_entities.pkl")
    
    ground_truth_path = os.path.join(base_dir, "processed_data/cadec_ground_truth.pkl")
    
    output_path = os.path.join(base_dir, "processed_data/verification_results.pkl")
    
    verification_system = EntityVerificationSystem()
    
    logging.info(f"Loading standardized entities from {standardized_path}...")
    standardized_df = pd.read_pickle(standardized_path)

    # Load ground truth annotations if available
    ground_truth_df = None
    if os.path.exists(ground_truth_path):
        logging.info(f"Loading ground truth annotations from {ground_truth_path}...")
        ground_truth_df = pd.read_pickle(ground_truth_path)
    else:
        logging.warning(f"Ground truth file not found at {ground_truth_path}")
    
    # Run verification
    overall_results = verification_system.verify_dataset(
        standardized_df=standardized_df,
        ground_truth_df=ground_truth_df,
        output_path=output_path
    )
    
    # Print summary
    print("\nVerification Summary:")
    print(f"Total posts: {overall_results['total_posts']}")
    print(f"Passed posts: {overall_results['passed_posts']} ({overall_results['pass_rate']:.2%})")
    print(f"Partial posts: {overall_results['partial_posts']} ({overall_results['partial_posts']/overall_results['total_posts']:.2%})")
    print(f"Failed posts: {overall_results['failed_posts']} ({overall_results['failure_rate']:.2%})")