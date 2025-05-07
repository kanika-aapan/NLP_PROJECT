import json
import os
import sys
import time
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import pandas as pd
import spacy

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

def plot_performance_vs_complexity(sentence_lengths, syntactic_depths, accuracies):
    """
    Plot system performance against sentence complexity.
    
    Args:
        sentence_lengths: List of sentence lengths
        syntactic_depths: List of syntactic parse tree depths
        accuracies: List of corresponding accuracies
    """
    plt.figure(figsize=(12, 8))
    
    # Calculate complexity score (combination of length and depth)
    complexity_scores = np.array(sentence_lengths) * 0.2 + np.array(syntactic_depths) * 0.8
    
    # Create scatter plot
    sc = plt.scatter(complexity_scores, accuracies, c=syntactic_depths, 
                    cmap='viridis', alpha=0.7, s=100)
    
    # Add trendline
    z = np.polyfit(complexity_scores, accuracies, 1)
    p = np.poly1d(z)
    plt.plot(complexity_scores, p(complexity_scores), "r--", alpha=0.8, 
             label=f"Trend: y={z[0]:.4f}x+{z[1]:.4f}")
    
    plt.colorbar(sc, label='Syntactic Depth')
    plt.xlabel('Sentence Complexity Score')
    plt.ylabel('Accuracy')
    plt.title('Performance vs. Sentence Complexity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('evaluation/complexity_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_classification(error_types, error_counts):
    """
    Create a horizontal bar chart of error types.
    
    Args:
        error_types: List of error type names
        error_counts: List of corresponding error counts
    """
    plt.figure(figsize=(12, 8))
    
    # Sort by frequency
    sorted_indices = np.argsort(error_counts)
    sorted_error_types = [error_types[i] for i in sorted_indices]
    sorted_error_counts = [error_counts[i] for i in sorted_indices]
    
    # Create horizontal bar chart
    bars = plt.barh(sorted_error_types, sorted_error_counts, color='maroon', alpha=0.8)
    
    # Add counts as text
    for i, (type_name, count) in enumerate(zip(sorted_error_types, sorted_error_counts)):
        plt.text(count + 0.5, i, str(count), va='center')
    
    plt.xlabel('Count')
    plt.ylabel('Error Type')
    plt.title('Classification of Error Types')
    plt.tight_layout()
    plt.savefig('evaluation/error_classification.png', dpi=300, bbox_inches='tight')
    plt.close()

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
    
    # Initialize spaCy for linguistic analysis
    try:
        nlp = spacy.load('en_core_web_md')
        print("Using spaCy model with word vectors")
    except:
        nlp = spacy.load('en_core_web_sm')
        print("Using basic spaCy model without word vectors")
    
    # Initialize metrics for sentence complexity
    sentence_lengths = []
    syntactic_depths = []
    accuracies_by_complexity = []
    
    # Initialize error classification
    error_types = [
        'Semantic Mismatch',       # When word meanings don't match correctly
        'Entity Not Found',        # When the system can't find an entity
        'Incorrect Relation',      # When the system matched the wrong relation
        'Ambiguous Question',      # When a question could have multiple interpretations
        'Missing Time/Place',      # When time/place constraints aren't met
        'Complex Sentence',        # When the sentence structure is too complex
        'Multiple Answers'         # When multiple possible answers exist
    ]
    error_counts = [0] * len(error_types)
    error_examples = [[] for _ in range(len(error_types))]  # To store examples
    
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
    
    # Group questions by complexity bins for later analysis
    complexity_bins = 5
    complexity_groups = {}
    for i in range(complexity_bins):
        complexity_groups[i] = {"correct": 0, "total": 0}
    
    # Test each question
    for qa_pair in tqdm(qa_pairs, desc="Evaluating questions"):
        question = qa_pair["question"]
        expected_answer = qa_pair["answer"].lower()
        
        # Calculate sentence complexity metrics
        doc = nlp(question)
        sentence_length = len(doc)
        sentence_lengths.append(sentence_length)
        
        # Calculate syntactic depth
        max_depth = 0
        for token in doc:
            # Count steps to root
            depth = 0
            current = token
            while current.head != current:  # While not at root
                depth += 1
                current = current.head
            max_depth = max(max_depth, depth)
        
        syntactic_depths.append(max_depth)
        
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
        else:
            # Classify the error
            if model_answer == "None":
                # Entity Not Found error
                error_counts[1] += 1
                error_examples[1].append((question, expected_answer, model_answer))
                
            elif sentence_length > 15 or max_depth > 5:
                # Complex Sentence error
                error_counts[5] += 1
                error_examples[5].append((question, expected_answer, model_answer))
                
            elif question_type == "when" and not any(t.like_num for t in nlp(model_answer)):
                # Missing Time error
                error_counts[4] += 1
                error_examples[4].append((question, expected_answer, model_answer))
                
            elif question_type == "where" and not any(t.ent_type_ in ["GPE", "LOC"] for t in nlp(model_answer) if t.ent_type_):
                # Missing Place error
                error_counts[4] += 1
                error_examples[4].append((question, expected_answer, model_answer))
                
            elif "," in expected_answer:
                # Possible Multiple Answers error
                error_counts[6] += 1
                error_examples[6].append((question, expected_answer, model_answer))
                
            elif any(relation_word in question.lower() for relation_word in ["relate", "connect", "link"]):
                # Incorrect Relation error
                error_counts[2] += 1
                error_examples[2].append((question, expected_answer, model_answer))
                
            elif any(q_word in question.lower() for q_word in ["can", "could", "possible", "might"]):
                # Ambiguous Question error
                error_counts[3] += 1
                error_examples[3].append((question, expected_answer, model_answer))
                
            else:
                # Default to Semantic Mismatch
                error_counts[0] += 1
                error_examples[0].append((question, expected_answer, model_answer))
        
        # Store result
        results.append({
            "question": question,
            "question_type": question_type,
            "expected_answer": expected_answer,
            "model_answer": model_answer,
            "is_correct": is_correct,
            "sentence_length": sentence_length,
            "syntactic_depth": max_depth
        })
        
        # Assign to complexity bin
        complexity_score = sentence_length * 0.2 + max_depth * 0.8
        max_complexity = 25 * 0.2 + 10 * 0.8  # Reasonable max values
        bin_index = min(int((complexity_score / max_complexity) * complexity_bins), complexity_bins - 1)
        
        complexity_groups[bin_index]["total"] += 1
        if is_correct:
            complexity_groups[bin_index]["correct"] += 1
    
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
    
    # Generate error classification visualization
    print("\nAnalyzing error patterns...")
    plot_error_classification(error_types, error_counts)
    
    # Save detailed error analysis
    with open("evaluation/error_analysis.txt", "w") as f:
        f.write("ERROR ANALYSIS REPORT\n")
        f.write("====================\n\n")
        f.write(f"Total errors: {sum(error_counts)} out of {len(results)}\n\n")
        
        for i, error_type in enumerate(error_types):
            f.write(f"\n{error_type.upper()} ERRORS ({error_counts[i]} instances)\n")
            f.write("-" * 50 + "\n\n")
            
            for j, (question, expected, actual) in enumerate(error_examples[i][:10]):  # Show up to 10 examples
                f.write(f"{j+1}. Question: {question}\n")
                f.write(f"   Expected: {expected}\n")
                f.write(f"   Received: {actual}\n\n")
    
    # Generate complexity vs performance visualization
    print("\nAnalyzing relationship between sentence complexity and accuracy...")
    
    # Calculate accuracy for each complexity group
    binned_complexities = []
    binned_accuracies = []
    
    for bin_index, counts in complexity_groups.items():
        if counts["total"] > 0:
            # Estimate the center value of this bin
            complexity_value = (bin_index + 0.5) * (max_complexity / complexity_bins)
            accuracy = counts["correct"] / counts["total"]
            
            binned_complexities.append(complexity_value)
            binned_accuracies.append(accuracy)
    
    # Get individual complexity scores for scatter plot
    complexity_scores = [sl * 0.2 + sd * 0.8 for sl, sd in zip(sentence_lengths, syntactic_depths)]
    # Create a binary accuracy list (1 for correct, 0 for incorrect)
    point_accuracies = [1.0 if r["is_correct"] else 0.0 for r in results]
    
    # Plot complexity vs performance
    plot_performance_vs_complexity(complexity_scores, syntactic_depths, point_accuracies)
    
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
    print(f"Error classification chart saved to evaluation/error_classification.png")
    print(f"Complexity performance chart saved to evaluation/complexity_performance.png")
    
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