import json
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from kwQnA._exportPairs import exportToJSON
from kwQnA._getentitypair import GetEntity
from kwQnA._qna import QuestionAnswer
from kwQnA._graph import GraphEnt

class KnowledgeGraphEvaluator:
    """Evaluator for Knowledge Graph based Question Answering system"""

    def __init__(self):
        self.getent = GetEntity()
        self.qa = QuestionAnswer()
        self.export = exportToJSON()
        self.graph = GraphEnt()
        
        # Make sure the evaluation directory exists
        if not os.path.exists(os.path.join(os.getcwd(), 'evaluation')):
            os.makedirs('evaluation')
    
    def process_evaluation_text(self, input_file):
        """Process evaluation text and create entity pairs"""
        print(f"Loading and processing text from {input_file}...")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                text_content = f.read()
        except:
            print(f"Error: Could not read file {input_file}")
            return None, 0
            
        start_time = time.time()
        refined_text = self.getent.preprocess_text([text_content])
        dataEntities, numberOfPairs = self.getent.get_entity(refined_text)
        
        if dataEntities:
            self.export.dumpdata(dataEntities[0])
            print(f"Created {numberOfPairs} entity pairs in {time.time() - start_time:.2f} seconds")
            
            # Save entity pairs for inspection
            dataEntities[0].to_csv('evaluation/entity_pairs.csv', index=False)
            print(f"Entity pairs saved to evaluation/entity_pairs.csv")
            
            # Optionally generate graph visualization
            if numberOfPairs < 500:  # Only generate graph if not too many entities
                try:
                    self.graph.createGraph(dataEntities[0])
                    print("Knowledge graph visualization created")
                except Exception as e:
                    print(f"Could not generate graph visualization: {e}")
            
            return dataEntities[0], numberOfPairs
        else:
            print("No entities extracted from text")
            return None, 0
    
    def evaluate_model(self, qa_file, number_of_pairs, output_file="evaluation/results.json"):
        """Evaluate the model on question-answer pairs from a JSON file"""
        try:
            with open(qa_file, 'r', encoding='utf-8') as f:
                qa_pairs = json.load(f)
            print(f"Loaded {len(qa_pairs)} question-answer pairs from {qa_file}")
        except:
            print(f"Error: Could not read question-answer pairs from {qa_file}")
            return None, None
        
        print("Starting evaluation...")
        total_questions = len(qa_pairs)
        correct_answers = 0
        results = []
        
        start_time = time.time()
        
        # Create categories for different question types
        categories = {
            "who": {"correct": 0, "total": 0},
            "what": {"correct": 0, "total": 0},
            "when": {"correct": 0, "total": 0},
            "where": {"correct": 0, "total": 0}
        }
        
        # Test each question
        for qa_pair in tqdm(qa_pairs, desc="Evaluating questions"):
            question = qa_pair["question"]
            expected_answer = qa_pair["answer"].lower()
            
            # Determine question type
            question_type = "other"
            if question.lower().startswith("who"):
                question_type = "who"
            elif question.lower().startswith("what"):
                question_type = "what"
            elif question.lower().startswith("when"):
                question_type = "when"
            elif question.lower().startswith("where"):
                question_type = "where"
            
            if question_type in categories:
                categories[question_type]["total"] += 1
            
            # Get model's answer
            model_answer = self.qa.findanswer(question, number_of_pairs)
            
            if model_answer is None or model_answer == "None" or model_answer == "Not Applicable":
                model_answer = "None"
            
            model_answer = str(model_answer).lower()
            
            # Check if answer is correct
            is_correct = False
            
            # Exact match
            if model_answer == expected_answer:
                is_correct = True
            # Partial match (model answer contains expected answer)
            elif expected_answer in model_answer:
                is_correct = True
            # Partial match (expected answer contains model answer)
            elif model_answer in expected_answer and model_answer != "none":
                is_correct = True
            
            if is_correct:
                correct_answers += 1
                if question_type in categories:
                    categories[question_type]["correct"] += 1
            
            # Store result
            results.append({
                "question": question,
                "question_type": question_type,
                "expected_answer": expected_answer,
                "model_answer": model_answer,
                "is_correct": is_correct
            })
        
        # Calculate overall accuracy
        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        
        # Calculate category accuracies
        category_accuracies = {}
        for category, data in categories.items():
            if data["total"] > 0:
                category_accuracies[category] = data["correct"] / data["total"]
            else:
                category_accuracies[category] = 0
        
        # Create summary
        summary = {
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "accuracy": accuracy,
            "category_accuracies": category_accuracies,
            "evaluation_time": time.time() - start_time
        }
        
        # Save detailed results
        with open(output_file, "w") as f:
            json.dump({
                "summary": summary,
                "results": results
            }, f, indent=2)
        
        print(f"\nEvaluation completed in {summary['evaluation_time']:.2f} seconds")
        print(f"Overall accuracy: {accuracy:.2%} ({correct_answers}/{total_questions})")
        print("\nAccuracy by question type:")
        for category, acc in category_accuracies.items():
            print(f"  {category.upper()}: {acc:.2%} ({categories[category]['correct']}/{categories[category]['total']})")
        
        print(f"\nDetailed results saved to {output_file}")
        
        # Create error analysis report
        self._create_error_analysis(results, "evaluation/error_analysis.txt")
        
        # Generate visualization
        self.plot_results(summary)
        
        return summary, results
    
    def _create_error_analysis(self, results, output_file):
        """Create error analysis report"""
        incorrect_results = [r for r in results if not r["is_correct"]]
        
        with open(output_file, "w") as f:
            f.write("ERROR ANALYSIS REPORT\n")
            f.write("===================\n\n")
            f.write(f"Total incorrect answers: {len(incorrect_results)}/{len(results)}\n\n")
            
            # Group by question type
            question_types = {}
            for r in incorrect_results:
                qt = r["question_type"]
                if qt not in question_types:
                    question_types[qt] = []
                question_types[qt].append(r)
            
            # Write grouped errors
            for qt, errors in question_types.items():
                f.write(f"\n{qt.upper()} QUESTIONS ({len(errors)} errors)\n")
                f.write("-" * 50 + "\n\n")
                
                for i, error in enumerate(errors, 1):
                    f.write(f"{i}. Question: {error['question']}\n")
                    f.write(f"   Expected: {error['expected_answer']}\n")
                    f.write(f"   Received: {error['model_answer']}\n\n")
        
        print(f"Error analysis report saved to {output_file}")
    
    def plot_results(self, summary, output_file="evaluation/accuracy.png"):
        """Plot evaluation results"""
        categories = list(summary["category_accuracies"].keys())
        accuracies = [summary["category_accuracies"][cat] for cat in categories]
        
        # Add overall accuracy
        categories.append("Overall")
        accuracies.append(summary["accuracy"])
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(categories, [acc * 100 for acc in accuracies])
        
        # Add the actual percentage on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.ylim(0, 100)  # Set y-axis to percentage scale
        plt.ylabel('Accuracy (%)')
        plt.title('Knowledge Graph QA System Accuracy by Question Type')
        plt.savefig(output_file)
        plt.close()
        
        print(f"Accuracy plot saved to {output_file}")

def main():
    """Main function to run the evaluation"""
    print("=" * 50)
    print("KNOWLEDGE GRAPH QA SYSTEM EVALUATION")
    print("=" * 50)
    
    # Create evaluator
    evaluator = KnowledgeGraphEvaluator()
    
    # Get input files
    eval_text_file = input("Enter path to evaluation text file: ")
    qa_pairs_file = input("Enter path to question-answer pairs file: ")
    
    # Process evaluation text
    entity_pairs, number_of_pairs = evaluator.process_evaluation_text(eval_text_file)
    
    if entity_pairs is not None:
        # Run evaluation
        summary, results = evaluator.evaluate_model(qa_pairs_file, number_of_pairs)
        
        # Print key insights
        if summary:
            print("\nKEY INSIGHTS:")
            print(f"- Overall accuracy: {summary['accuracy']:.2%}")
            
            # Best performing question type
            best_qt = max(summary["category_accuracies"].items(), key=lambda x: x[1])
            print(f"- Best performing question type: {best_qt[0].upper()} questions ({best_qt[1]:.2%})")
            
            # Worst performing question type
            worst_qt = min(summary["category_accuracies"].items(), key=lambda x: x[1])
            print(f"- Worst performing question type: {worst_qt[0].upper()} questions ({worst_qt[1]:.2%})")
            
            # Completion time
            print(f"- Evaluation completed in {summary['evaluation_time']:.2f} seconds")

if __name__ == "__main__":
    main()