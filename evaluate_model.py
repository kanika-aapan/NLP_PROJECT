import json
import os
import sys
import time
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from kwQnA._exportPairs import exportToJSON
from kwQnA._getentitypair import GetEntity
from kwQnA._qna import QuestionAnswer
from kwQnA._graph import GraphEnt

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Knowledge Graph QA System')
    
    parser.add_argument('--text', type=str, required=True,
                        help='Path to evaluation text file')
    
    parser.add_argument('--qa', type=str, required=True,
                        help='Path to question-answer pairs JSON file')
    
    parser.add_argument('--output', type=str, default='evaluation/results.json',
                        help='Path to output results file (default: evaluation/results.json)')
    
    parser.add_argument('--graph', action='store_true',
                        help='Generate knowledge graph visualization (may be slow for large datasets)')
    
    parser.add_argument('--detailed', action='store_true',
                        help='Generate detailed error analysis')
    
    return parser.parse_args()

def load_file(file_path):
    """Load file contents"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        sys.exit(1)

def create_output_dir(output_path):
    """Create output directory if it doesn't exist"""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

def evaluate_model():
    """Run evaluation on the knowledge graph QA system"""
    args = parse_arguments()
    
    # Create output directory
    create_output_dir(args.output)
    
    # Load evaluation text
    print(f"Loading evaluation text from {args.text}...")
    text_content = load_file(args.text)
    
    # Load question-answer pairs
    print(f"Loading question-answer pairs from {args.qa}...")
    try:
        qa_pairs = json.loads(load_file(args.qa))
        print(f"Loaded {len(qa_pairs)} question-answer pairs")
    except json.JSONDecodeError:
        print(f"Error: {args.qa} is not a valid JSON file")
        sys.exit(1)
    
    # Initialize components
    getent = GetEntity()
    qa = QuestionAnswer()
    export = exportToJSON()
    
    # Process evaluation text
    print("Processing evaluation text and extracting entity pairs...")
    start_time = time.time()
    refined_text = getent.preprocess_text([text_content])
    dataEntities, numberOfPairs = getent.get_entity(refined_text)
    
    if not dataEntities:
        print("Error: Failed to extract entities from text")
        sys.exit(1)
    
    # Export entity pairs
    export.dumpdata(dataEntities[0])
    print(f"Created {numberOfPairs} entity pairs in {time.time() - start_time:.2f} seconds")
    
    # Optionally generate graph visualization
    if args.graph:
        print("Generating knowledge graph visualization...")
        try:
            graph = GraphEnt()
            graph.createGraph(dataEntities[0])
            print("Knowledge graph visualization created")
        except Exception as e:
            print(f"Warning: Could not generate graph visualization: {e}")
    
    # Run evaluation
    print("\nStarting evaluation...")
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
        model_answer = qa.findanswer(question, numberOfPairs)
        
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
    
    evaluation_time = time.time() - start_time
    
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
        "evaluation_time": evaluation_time
    }
    
    # Save detailed results
    with open(args.output, "w") as f:
        json.dump({
            "summary": summary,
            "results": results
        }, f, indent=2)
    
    # Plot accuracy chart
    plot_results(summary)
    
    # Generate error analysis if requested
    if args.detailed:
        create_error_analysis(results)
    
    # Print summary
    print(f"\nEvaluation completed in {evaluation_time:.2f} seconds")
    print(f"Overall accuracy: {accuracy:.2%} ({correct_answers}/{total_questions})")
    print("\nAccuracy by question type:")
    for category, acc in category_accuracies.items():
        total = categories[category]["total"]
        correct = categories[category]["correct"]
        if total > 0:
            print(f"  {category.upper()}: {acc:.2%} ({correct}/{total})")
    
    print(f"\nDetailed results saved to {args.output}")
    print(f"Accuracy chart saved to evaluation/accuracy.png")
    
    if args.detailed:
        print(f"Error analysis saved to evaluation/error_analysis.txt")
    
    return summary, results

def plot_results(summary):
    """Plot evaluation results"""
    # Create output directory if it doesn't exist
    if not os.path.exists('evaluation'):
        os.makedirs('evaluation')
    
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
    plt.savefig("evaluation/accuracy.png")
    plt.close()

def create_error_analysis(results):
    """Create error analysis report"""
    # Create output directory if it doesn't exist
    if not os.path.exists('evaluation'):
        os.makedirs('evaluation')
    
    incorrect_results = [r for r in results if not r["is_correct"]]
    
    with open("evaluation/error_analysis.txt", "w") as f:
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

if __name__ == "__main__":
    evaluate_model()