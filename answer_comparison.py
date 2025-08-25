import json
import os
from dotenv import load_dotenv

# LLM APIs
import google.generativeai as genai
import openai
import anthropic

# Load environment variables
load_dotenv(dotenv_path="config/.env")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# Load API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

# Load prompt template from JSON
def load_prompt_template():
    with open("prompts/2_prompt.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts["java_mcq_prompt_template"]

PROMPT_TEMPLATE = load_prompt_template()

# Validate and setup provider
if LLM_PROVIDER == "gemini":
    if not GEMINI_API_KEY:
        raise EnvironmentError("❌ Gemini API key missing.")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")

elif LLM_PROVIDER == "chatgpt":
    if not OPENAI_API_KEY:
        raise EnvironmentError("❌ OpenAI API key missing.")
    openai.api_key = OPENAI_API_KEY

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
    if not CLAUDE_API_KEY:
        raise EnvironmentError("❌ Claude API key missing.")
    claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

else:
    raise ValueError(f"❌ Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. Use gemini/chatgpt/azure/claude.")


def normalize(text: str) -> str:
    return text.strip().lower() if text else ""


def ask_model(code: str, question_text: str, options: list[str]) -> str:
    formatted_options = "\n".join([f"{i + 1}. {opt}" for i, opt in enumerate(options)])
    prompt = PROMPT_TEMPLATE.format(code=code, question=question_text, options=formatted_options)

    try:
        if LLM_PROVIDER == "gemini":
            response = model.generate_content(prompt)
            answer = response.text.strip()

        elif LLM_PROVIDER == "chatgpt":
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a Java programming expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            answer = response.choices[0].message.content.strip()

        elif LLM_PROVIDER == "azure":
            response = openai.ChatCompletion.create(
                deployment_id=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are a Java programming expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            answer = response.choices[0].message.content.strip()

        elif LLM_PROVIDER == "claude":
            response = claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                temperature=0.2,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.content[0].text.strip()

        for opt in options:
            if normalize(answer) == normalize(opt):
                return opt
        return f"[Unmatched] {answer}"

    except Exception as e:
        return f"❌ Error: {e}"


def validate_attempts(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        attempts_data = json.load(f)

    results = []

    for attempt in attempts_data:
        attempt_result = {
            "attempt_id": attempt["attempt_id"],
            "student_username": attempt["student_username"],
            "student_email": attempt["student_email"],
            "questions": []
        }

        code = attempt["code_used_to_generate"]

        for q in attempt["questions"]:
            question_text = q["question_text"]
            options = q["options"]
            correct = q["correct_answer_text"]
            student = q.get("student_answer_text")

            llm_answer = ask_model(code, question_text, options)

            result = {
                "question_id": q["question_id"],
                "question_text": question_text,
                f"{LLM_PROVIDER}_answer": llm_answer,
                "correct_answer_text": correct,
                "student_answer_text": student,
                f"is_{LLM_PROVIDER}_correct": normalize(llm_answer) == normalize(correct),
                f"does_student_match_{LLM_PROVIDER}": normalize(llm_answer) == normalize(student) if student else False
            }

            attempt_result["questions"].append(result)

        results.append(attempt_result)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump(results, out_file, indent=2, ensure_ascii=False)

    print(f"✅ Results saved to: {output_path}")


# Run it
if __name__ == "__main__":
    input_json = "input_data/minimal_data/detailed_question_data.json"
    output_json = f"json_reports/comparison_{LLM_PROVIDER}_metrics_report.json"
    validate_attempts(input_json, output_json)

