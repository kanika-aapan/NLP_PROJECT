from datetime import datetime
import json
import os

from flask import Flask, jsonify, redirect, render_template, request, url_for, send_file
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from kwQnA._exportPairs import exportToJSON
from kwQnA._getentitypair import GetEntity
from kwQnA._qna import QuestionAnswer
from kwQnA._graph import GraphEnt

app = Flask(__name__)


class CheckAndSave:
    """docstring for CheckAndSave."""

    def __init__(self):
        super(CheckAndSave, self).__init__()

    def createdataset(self, para, que, ent, ans1, ans2):

        wholedata = {"para":[str(para)],"que":[[str(que)]], "entities":[ent], "ans1": [ans1], "ans2":[ans2]}
        # print(wholedata)
        # return None

class OurModel:
    def __init__(self):
        self.getent = GetEntity()
        self.qa = QuestionAnswer()
        self.export = exportToJSON()

    def getAnswer(self, paragraph, question):

            with open("temp_eval.txt", "w", encoding="utf-8") as temp_file:
                temp_file.write(text_content)

            with open("temp_eval.txt", "r", encoding="utf-8") as temp_file:
                lines = temp_file.readlines()
        
            refined_text = getent.nlp(text_content)  # Just use spaCy directly
            dataEntities, numberOfPairs = getent.get_entity(refined_text)
            dataEntities, numberOfPairs = self.getent.get_entity(refined_text)

            if dataEntities:
                # data_in_dict = dataEntities[0].to_dict()
                self.export.dumpdata(dataEntities[0])
                outputAnswer = self.qa.findanswer(str(question), numberOfPairs)
                if outputAnswer == []:
                    return None
                return outputAnswer
            return None


# Evaluation class for integrating evaluation functionality
class Evaluator:
    def __init__(self):
        self.getent = GetEntity()
        self.qa = QuestionAnswer()
        self.export = exportToJSON()
        self.graph = GraphEnt()
        
        # Make sure the evaluation directory exists
        if not os.path.exists(os.path.join(os.getcwd(), 'evaluation')):
            os.makedirs('evaluation')
    
    def process_evaluation_text(self, text_content):
        """Process evaluation text and create entity pairs"""
        with open("temp_eval.txt", "w", encoding="utf-8") as temp_file:
            temp_file.write(text_content)

        with open("temp_eval.txt", "r", encoding="utf-8") as temp_file:
            lines = temp_file.readlines()
    
        refined_text = getent.nlp(text_content)  # Just use spaCy directly
        dataEntities, numberOfPairs = getent.get_entity(refined_text)
        dataEntities, numberOfPairs = self.getent.get_entity(refined_text)
        
        if dataEntities:
            self.export.dumpdata(dataEntities[0])
            return dataEntities[0], numberOfPairs
        else:
            return None, 0
    
    def evaluate_model(self, qa_pairs, number_of_pairs):
        """Evaluate the model on question-answer pairs"""
        total_questions = len(qa_pairs)
        correct_answers = 0
        results = []
        
        # Create categories for different question types
        categories = {
            "who": {"correct": 0, "total": 0},
            "what": {"correct": 0, "total": 0},
            "when": {"correct": 0, "total": 0},
            "where": {"correct": 0, "total": 0}
        }
        
        # Test each question
        for qa_pair in qa_pairs:
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
            
            if model_answer is None:
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
            "category_accuracies": category_accuracies
        }
        
        # Save detailed results
        with open("evaluation/results.json", "w") as f:
            json.dump({
                "summary": summary,
                "results": results
            }, f, indent=2)
        
        # Create visualization
        self._plot_results(summary)
        
        return summary, results
    
    def _plot_results(self, summary):
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
        plt.savefig("evaluation/accuracy.png")
        plt.close()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=20550, threaded=True)