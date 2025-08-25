# main_script.py
import json
import re
import os
import time
from dotenv import load_dotenv
from prompt_handler import PromptManager

# --- Load Configuration ---
load_dotenv(dotenv_path='config/.env')
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# --- Initialize Prompt Manager ---
prompt_manager = PromptManager()

# --- Initialize LLM Based on Provider ---
if LLM_PROVIDER == "gemini":
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found.")
    genai.configure(api_key=GEMINI_API_KEY)
    MODEL = genai.GenerativeModel('gemini-2.5-flash')

elif LLM_PROVIDER == "chatgpt":
    import openai
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found.")
    openai.api_key = OPENAI_API_KEY
    MODEL = "gpt-4"

# elif LLM_PROVIDER == "azure":
#     import openai
#     AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
#     AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
#     AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
#     AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-03-01-preview")

#     if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT:
#         raise ValueError("Azure OpenAI environment variables not properly configured.")

#     openai.api_type = "azure"
#     openai.api_key = AZURE_OPENAI_KEY
#     openai.api_base = AZURE_OPENAI_ENDPOINT
#     openai.api_version = AZURE_OPENAI_API_VERSION
#     MODEL = AZURE_OPENAI_DEPLOYMENT  # deployment_id for Azure

elif LLM_PROVIDER == "azure"  and all(k in os.environ for k in [
    "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_VERSION"
]):
    import openai
    openai.api_type = "azure"
    openai.api_base = os.environ["AZURE_OPENAI_ENDPOINT"]
    openai.api_key = os.environ["AZURE_OPENAI_API_KEY"]
    openai.api_version = os.environ["AZURE_OPENAI_API_VERSION"]
    AZURE_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    MODEL = AZURE_OPENAI_DEPLOYMENT   # <-- Add this line
    USE_AZURE_OPENAI = True
    print("✅ Azure OpenAI API configured.")
    

elif LLM_PROVIDER == "claude":
    try:
        import anthropic
    except ImportError:
        raise ImportError("The 'anthropic' module is missing. Install it using 'pip install anthropic'")
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
    if not CLAUDE_API_KEY:
        raise ValueError("CLAUDE_API_KEY not found.")
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

else:
    raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")

# --- Paths and Constants ---
JSON_FILE_PATH = 'json_reports/code_topics_report.json'
OUTPUT_RESULTS_FILE = f'json_reports/2switchmodel_{LLM_PROVIDER}_metrics_report.json'
dynamic_topic_keywords = {}

# --- LLM Unified Function ---
def generate_response(prompt: str) -> str:
    time.sleep(0.5)
    try:
        if LLM_PROVIDER == "gemini":
            response = MODEL.generate_content(prompt)
            return response.text.strip()

        elif LLM_PROVIDER == "chatgpt":
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            return response['choices'][0]['message']['content'].strip()

        elif LLM_PROVIDER == "azure":
            response = openai.ChatCompletion.create(
                deployment_id=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            return response['choices'][0]['message']['content'].strip()

        elif LLM_PROVIDER == "claude":
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                temperature=0.5,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()

    except Exception as e:
        print(f"Error from {LLM_PROVIDER} while generating response: {e}")
        return "LLM error occurred."

# --- Functional Wrappers ---
def ask_llm_yes_no(prompt: str) -> bool:
    response_text = generate_response(prompt).lower()
    if 'yes' in response_text:
        return True
    elif 'no' in response_text:
        return False
    print(f"Warning: Unexpected Yes/No response: '{response_text}'")
    return False

def explain_question_code_relevance(question_text: str, code_snippet: str) -> str:
    prompt = prompt_manager.get("explain_code_relevance", question=question_text, code=code_snippet)
    return generate_response(prompt)

def generate_keywords_with_llm(topic: str) -> list:
    print(f"--- Calling LLM for keywords for topic: '{topic}' ---")
    prompt = prompt_manager.get("generate_keywords", topic=topic)
    response_text = generate_response(prompt).lower()
    return [k.strip() for k in response_text.split(',') if k.strip()]

def get_or_generate_keywords(topic: str) -> list:
    if topic not in dynamic_topic_keywords:
        try:
            dynamic_topic_keywords[topic] = generate_keywords_with_llm(topic)
        except Exception:
            dynamic_topic_keywords[topic] = [topic.lower().replace(" ", ""), "method", "class", "variable"]
    return dynamic_topic_keywords[topic]

def verify_topics_in_code(code: str, topics: list) -> dict:
    code_lower = code.lower()
    verification_results = {}
    match = re.search(r"public\\s+class\\s+(\\w+)", code_lower)
    class_name = match.group(1).lower() if match else None

    for topic in topics:
        found = False
        keywords = get_or_generate_keywords(topic)

        if "constructor" in topic.lower() and class_name:
            patterns = [f"public {class_name}(", f"new {class_name}("]
            found = any(p in code_lower for p in patterns) or any(k in code_lower for k in keywords)
        else:
            found = any(k in code_lower for k in keywords)

        verification_results[topic] = "Relevant" if found else "Not Clearly Relevant (Keyword not found)"
    return verification_results

def check_question_to_code_relevance(question_text: str, code_snippet: str) -> bool:
    prompt = prompt_manager.get("question_to_code_yesno", question=question_text, code=code_snippet)
    return ask_llm_yes_no(prompt)

# --- Main Execution ---
print(f"Using LLM Provider: {LLM_PROVIDER}")
print(f"Current Working Directory: {os.getcwd()}")

if not os.path.exists(JSON_FILE_PATH):
    print(f"Error: Input JSON file '{JSON_FILE_PATH}' not found.")
    exit()

try:
    with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
except Exception as e:
    print(f"Failed to load JSON file: {e}")
    exit()

all_unique_topics = set()
for attempt in data:
    all_unique_topics.update(attempt.get("inferred_code_topics", []))

print("\n=== Step 1: Keyword Generation ===\n")
for topic in all_unique_topics:
    _ = get_or_generate_keywords(topic)
    print(f"  {topic}: {dynamic_topic_keywords[topic]}")

print("\n=== Step 2: Analyzing Student Attempts ===\n")
results_output_data = []
correct_flags = 0
total_flagged = 0

for attempt in data:
    attempt_id = attempt.get("attempt_id", "N/A")
    student_username = attempt.get("student_username", "N/A")
    code = attempt.get("code_used_to_generate", "")
    inferred_topics = attempt.get("inferred_code_topics", [])
    generated_questions = attempt.get("generated_questions", [])

    print(f"\n--- Attempt ID: {attempt_id} | Student: {student_username} ---")

    result = {
        "attempt_id": attempt_id,
        "student_username": student_username,
        "student_email": attempt.get("student_email", "N/A"),
        "inferred_code_topics": inferred_topics,
        "code_topic_verification_results": {},
        "question_analysis": []
    }

    if not code:
        for topic in inferred_topics:
            result["code_topic_verification_results"][topic] = "No Code to Verify"
    else:
        topic_verification = verify_topics_in_code(code, inferred_topics)
        result["code_topic_verification_results"] = topic_verification
        for topic, status in topic_verification.items():
            print(f"  - {topic}: {status}")

    for question in generated_questions:
        q_id = question.get("question_id", "N/A")
        q_text = question.get("question_text", "N/A")
        original_flag = question.get("this_doesnt_seem_right", False)

        analysis = {
            "question_id": q_id,
            "question_text": q_text,
            "original_this_doesnt_seem_right": original_flag,
            "genai_question_to_code_relevance": None,
            "relevance_explanation": "",
            "this_doesnt_seem_right_verification_status": "",
            "student_flag_correct": False
        }

        if not code:
            analysis.update({
                "genai_question_to_code_relevance": "No Code to Verify",
                "relevance_explanation": "No code provided, so relevance could not be assessed.",
                "this_doesnt_seem_right_verification_status": "Not Applicable (No code)",
                "student_flag_correct": False
            })
        else:
            is_relevant = check_question_to_code_relevance(q_text, code)
            explanation = explain_question_code_relevance(q_text, code)

            analysis["genai_question_to_code_relevance"] = is_relevant
            analysis["relevance_explanation"] = explanation

            if is_relevant:
                if not original_flag:
                    analysis["this_doesnt_seem_right_verification_status"] = "✅ Correct (Student marked relevant, and it is relevant)"
                    correct_flags += 1
                    analysis["student_flag_correct"] = True
                else:
                    analysis["this_doesnt_seem_right_verification_status"] = "❌ Incorrect (Student marked NOT relevant, but it is relevant)"
            else:
                if original_flag:
                    analysis["this_doesnt_seem_right_verification_status"] = "✅ Correct (Student marked NOT relevant, and it is NOT relevant)"
                    correct_flags += 1
                    analysis["student_flag_correct"] = True
                else:
                    analysis["this_doesnt_seem_right_verification_status"] = "❌ Incorrect (Student marked relevant, but it is NOT relevant)"
            total_flagged += 1

        result["question_analysis"].append(analysis)

    results_output_data.append(result)

summary_stats = {
    "total_questions_flagged": total_flagged,
    "correct_flags": correct_flags,
    "accuracy_percentage": round((correct_flags / total_flagged) * 100, 2) if total_flagged > 0 else 0.0
}

# --- Save Results ---
print("\nSaving results...")
try:
    output_payload = {
        "summary_statistics": summary_stats,
        "analysis_results": results_output_data
    }
    with open(OUTPUT_RESULTS_FILE, 'w', encoding='utf-8') as out:
        json.dump(output_payload, out, indent=2)
    print(f"Results saved to {OUTPUT_RESULTS_FILE}")
except Exception as e:
    print(f"Error saving results: {e}")

if total_flagged > 0:
    print(f"\n=== Summary ===")
    print(f"Total Questions Analyzed: {total_flagged}")
    print(f"Correct Student Flags: {correct_flags}")
    print(f"Accuracy of Student Flags: {summary_stats['accuracy_percentage']:.2f}%")
else:
    print("\nNo questions had flags to analyze.")
