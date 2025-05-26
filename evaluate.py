from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import os
import json

def label_score(input_dir):
    result_file = os.path.join(input_dir, "label_result.txt")
    with open(result_file, 'w', encoding='utf-8') as result_txt:
	categories = [ "price", "relevance", "math"]
        accuracies = []
        for cate in categories:
            all_true_labels = []
            all_pred_labels = []
            yes = 0
            total = 0
            tasks_file = os.path.join(input_dir, f"{cate}_tasks.json")
            print(tasks_file)
            with open(tasks_file, 'r', encoding='utf-8') as f:
                tasks_data = json.load(f)

            for task in tasks_data:
                LLM_optimal = task.get("LLMAnswer", {}).get("CandidateAnswer", None)
                GT_optimal = task.get("Groundtruth", {}).get("ReferenceAnswer", None)
                all_true_labels.append(GT_optimal)
                all_pred_labels.append(LLM_optimal)
                if GT_optimal == LLM_optimal:
                    yes+=1
                total+=1

            if cate == "math":
                all_pred_labels = [label if label is not None else "0" for label in all_pred_labels]
            elif cate == "relevance":
                all_pred_labels = [label if label is not None else "-1(No Result)" for label in all_pred_labels]
            else:
                all_pred_labels = [label if label is not None else -1 for label in all_pred_labels]
                all_pred_labels = [label if label is not '' else -1 for label in all_pred_labels]

            print(f"accuracy is {yes} vs {total}...{yes/total}"),
            print(all_true_labels)
            print(all_pred_labels)

            # calculate Precision, Recall, F1-score and Accuracy
            precision = precision_score(all_true_labels, all_pred_labels, average='macro', zero_division=0)
            recall = recall_score(all_true_labels, all_pred_labels, average='macro', zero_division=0)
            f1 = f1_score(all_true_labels, all_pred_labels, average='macro', zero_division=0)
            accuracy = accuracy_score(all_true_labels, all_pred_labels)
            accuracies.append(accuracy)

            # output in a txt file
            result_txt.write(f"Category: {cate}\n")
            result_txt.write(f"Precision: {precision:.4f}\n")
            result_txt.write(f"Recall: {recall:.4f}\n")
            result_txt.write(f"F1-score: {f1:.4f}\n")
            result_txt.write(f"Accuracy: {accuracy:.4f}\n")
            result_txt.write("==========\n")

	# multiple choice tasks
        categories = ["chat", "summary", "hallu"] 
        for cate in categories:
            all_true_labels = []
            all_pred_labels = []
            yes = 0
            total = 0
            tasks_file = os.path.join(input_dir, f"{cate}_tasks.json")
            print(tasks_file)

            with open(tasks_file, 'r', encoding='utf-8') as f:
                tasks_data = json.load(f)

            for task in tasks_data:
                LLM_optimal = task.get("LLMAnswer", {}).get("OptimalOption", "E")
                GT_optimal = task.get("Groundtruth", {}).get("OptimalOption", "E")
                all_true_labels.append(GT_optimal)
                all_pred_labels.append(LLM_optimal)
                if GT_optimal == LLM_optimal:
                    yes+=1
                total+=1
            all_pred_labels = [label if type(label) is str else "E" for label in all_pred_labels]
            
            # compute Precision, Recall, F1-score and Accuracy
            precision = precision_score(all_true_labels, all_pred_labels, average='macro', zero_division=0)
            recall = recall_score(all_true_labels, all_pred_labels, average='macro', zero_division=0)
            f1 = f1_score(all_true_labels, all_pred_labels, average='macro', zero_division=0)
            accuracy = accuracy_score(all_true_labels, all_pred_labels)
            accuracies.append(accuracy)

             # output in a txt file
            result_txt.write(f"Category: {cate}\n")
            result_txt.write(f"Precision: {precision:.4f}\n")
            result_txt.write(f"Recall: {recall:.4f}\n")
            result_txt.write(f"F1-score: {f1:.4f}\n")
            result_txt.write(f"Accuracy: {accuracy:.4f}\n")
            result_txt.write("========\n")

        task_counts = [442, 192, 52, 180, 58, 118]
        weighted_accuracy = sum(a * t for a, t in zip(accuracies, task_counts)) / sum(task_counts)

        print(f"weighted acc: {weighted_accuracy:.4f}")
        result_txt.write(f"all: {weighted_accuracy:.4f}\n\n")


if __name__ == "__main__":
    # wholes = [
    #         "./final/deepseek_v3_answer",
    #           "./final/Qwen2_5_7B_answer", 
    #           "./final/deepseek_r1_answer", 
    #           "./final/abation_4o_answer",
    #           "./final/4o_answer",
    #           "./final/R1_Distill_Qwen_7B_answer", 
    #           "./final/o1_answer",
    #           "./final/Llama3_8B_answer", 
    #           "./final/Vicuna_7B_answer", 
    #           "./final/Qwen2_5_7B_Instruct_answer",
    #           "./final/Qwen2_5_32B_answer",
    #           "./final/R1_Distill_Qwen_32B_answer"
    #           ]
    # for input_dir in wholes:
    #     label_score(input_dir)
    #     print('==*************************===')
    input_dir = "YOUR_ANSWER_PATH"
    label_score(input_dir)