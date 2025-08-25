import json
import os

# File paths - adjust as needed
INPUT_FILE = "input_data/bulk_data/raw_dataset.json"
RAGAS_OUTPUT_FILE = "input_data/minimal_data/formatted_ragas_data_extra.json"
DEEPEVAL_OUTPUT_FILE = "input_data/minimal_data/formatted_deepeval_data_extra.json"
SUMMARY_OUTPUT_FILE = "input_data/minimal_data/summary_data.json"
DETAILED_OUTPUT_FILE = "input_data/minimal_data/detailed_question_data.json"
DEEPEVAL_FROM_DETAILED_OUTPUT = "input_data/minimal_data/deepeval_from_detailed.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved JSON data to '{path}'")
    except IOError as e:
        print(f"❌ Error writing to '{path}': {e}")

def extract_formatted_entries(data):
    ragas_formatted = []
    deepeval_formatted = []

    def process_attempt(attempt):
        context = attempt.get("CodeUsedToGenerate", "").replace("\r", "")
        for q in attempt.get("questions", []):
            correct = q.get("Answer", {}).get("optionText", "").strip()
            student = q.get("StudentAnswer", {}).get("optionText", "").strip() if q.get("StudentAnswer") else ""
            question = q.get("Question", "").strip()

            if question and correct:
                # RAGAS format
                ragas_formatted.append({
                    "question": question,
                    "ground_truths": [correct],
                    "reference": correct,
                    "contexts": [context],
                    "answer": correct,
                    "student_answer": student
                })

                # DeepEval format
                deepeval_formatted.append({
                    "input": question,
                    "actual_output": student,
                    "expected_output": correct,
                    "context": [context] if context else []
                })

    attempts = []
    if isinstance(data, dict) and "data" in data:
        attempts = data["data"]
    elif isinstance(data, list):
        attempts = data
    else:
        print("⚠️ Warning: Unexpected data structure, no attempts found.")

    for attempt in attempts:
        process_attempt(attempt)

    return ragas_formatted, deepeval_formatted

def extract_and_store_summary(attempts, summary_output_path):
    extracted_results = []

    for attempt in attempts:
        extracted_results.append({
            "attempt_id": attempt.get("ID", "N/A"),
            "student_username": attempt.get("studentUsername", "N/A"),
            "student_email": attempt.get("studentEmail", "N/A"),
            "code_used_to_generate": attempt.get("CodeUsedToGenerate", "").strip() or "No code available.",
            "generated_questions": [
                {
                    "question_id": q.get("ID", "N/A"),
                    "question_text": q.get("Question", "No question text."),
                    "this_doesnt_seem_right": q.get("this_doesnt_seem_right", False)
                }
                for q in attempt.get("questions", [])
            ]
        })

    save_json(extracted_results, summary_output_path)

def extract_and_store_detailed(attempts, detailed_output_path):
    output_data = []

    for attempt in attempts:
        attempt_entry = {
            "attempt_id": attempt.get("ID", "N/A"),
            "student_username": attempt.get("studentUsername", "N/A"),
            "student_email": attempt.get("studentEmail", "N/A"),
            "code_used_to_generate": attempt.get("CodeUsedToGenerate", "").strip() or "No code provided.",
            "questions": []
        }

        for question in attempt.get("questions", []):
            question_entry = {
                "question_id": question.get("ID", "N/A"),
                "question_text": question.get("Question", "No question provided."),
                "correct_answer_text": question.get("Answer", {}).get("optionText", "No correct answer."),
                "options": [opt.get("optionText", "N/A") for opt in question.get("Options", [])],
                "student_answer_text": question.get("StudentAnswer", {}).get("optionText") if question.get("StudentAnswer") else None,
                "this_doesnt_seem_right": question.get("this_doesnt_seem_right", False)
            }
            attempt_entry["questions"].append(question_entry)

        output_data.append(attempt_entry)

    save_json(output_data, detailed_output_path)

def format_for_deepeval(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    formatted_data = []

    for entry in raw_data:
        code_context = entry.get("code_used_to_generate", "")
        questions = entry.get("questions", [])

        for q in questions:
            question_text = q.get("question_text", "").strip()
            correct_answer = q.get("correct_answer_text", "").strip()
            student_answer = (q.get("student_answer_text") or "").strip()

            formatted_entry = {
                "input": question_text,
                "actual_output": student_answer,
                "expected_output": correct_answer,
                "context": code_context
            }
            formatted_data.append(formatted_entry)

    save_json(formatted_data, output_path)

def generate_additional_ragas_datasets(detailed_input_path):
    """
    Generate 4 RAGAS datasets from detailed_question_data.json,
    saving them as separate files.
    """
    output_dir = os.path.dirname(detailed_input_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(detailed_input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    ragas_datasets = {
        "q1_question_vs_code": [],
        "q2_option_vs_question_code": [],
        "q3_correct_answer_vs_question": [],
        "q4_student_vs_llm_answer": []
    }

    for attempt in raw_data:
        code = attempt.get("code_used_to_generate", "")
        for q in attempt.get("questions", []):
            question = (q.get("question_text") or "").strip()
            correct = (q.get("correct_answer_text") or "").strip()
            student = (q.get("student_answer_text") or "").strip()
            options = q.get("options", [])

            # Q1: Question vs Code
            if question and code:
                ragas_datasets["q1_question_vs_code"].append({
                    "question": question,
                    "answer": question,
                    "reference": question,
                    "contexts": [code]
                })

            # Q2: Options vs Question+Code
            for opt in options:
                full_context = f"{code}\n\nQ: {question}"
                ragas_datasets["q2_option_vs_question_code"].append({
                    "question": opt,
                    "answer": opt,
                    "reference": correct,
                    "contexts": [full_context]
                })

            # Q3: Correct Answer vs Question
            if question and correct:
                ragas_datasets["q3_correct_answer_vs_question"].append({
                    "question": correct,
                    "answer": correct,
                    "reference": question,
                    "contexts": [question]
                })

            # Q4: Student Answer vs Correct Answer
            if correct and student:
                ragas_datasets["q4_student_vs_llm_answer"].append({
                    "question": correct,
                    "answer": student,
                    "reference": correct,
                    "contexts": [code]
                })

    # Save datasets to files
    for key, dataset in ragas_datasets.items():
        file_path = os.path.join(output_dir, f"formatted_ragas_{key}.json")
        save_json(dataset, file_path)

    print("✅ All 4 RAGAS dataset files generated inside the detailed data folder.")

def main():
    # Load input data once
    raw_data = load_json(INPUT_FILE)

    # Extract attempts list from input
    attempts = []
    if isinstance(raw_data, dict) and "data" in raw_data:
        attempts = raw_data["data"]
    elif isinstance(raw_data, list):
        attempts = raw_data
    else:
        print("❌ Input JSON does not contain expected 'data' field or list of attempts.")
        return

    # 1. Extract for RAGAS & DeepEval (direct)
    ragas_data, deepeval_data = extract_formatted_entries(raw_data)
    save_json(ragas_data, RAGAS_OUTPUT_FILE)
    save_json(deepeval_data, DEEPEVAL_OUTPUT_FILE)

    # 2. Extract and save summary and detailed JSONs
    extract_and_store_summary(attempts, SUMMARY_OUTPUT_FILE)
    extract_and_store_detailed(attempts, DETAILED_OUTPUT_FILE)

    # 3. Format detailed JSON for DeepEval
    format_for_deepeval(DETAILED_OUTPUT_FILE, DEEPEVAL_FROM_DETAILED_OUTPUT)

    # 4. Generate the 4 additional RAGAS datasets from detailed JSON
    generate_additional_ragas_datasets(DETAILED_OUTPUT_FILE)

if __name__ == "__main__":
    main()
