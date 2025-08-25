import json
import os
from dotenv import load_dotenv
from prompt_handler import PromptManager
import openai
# --- Load environment variables ---
load_dotenv(dotenv_path='config/.env')
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# --- Flags and Clients ---
USE_GEMINI = USE_OPENAI = USE_CLAUDE = USE_AZURE_OPENAI = False
GEMINI_MODEL = None
OPENAI_MODEL_NAME = "gpt-3.5-turbo"
CLAUDE_MODEL_NAME = "claude-3-sonnet-20240229"
CLAUDE_CLIENT = None
AZURE_OPENAI_DEPLOYMENT = None

# --- Prompt Manager ---
prompt_mgr = PromptManager()

# --- LLM Setup Function ---
def setup_llm():
    global USE_GEMINI, USE_OPENAI, USE_CLAUDE, USE_AZURE_OPENAI
    global GEMINI_MODEL, CLAUDE_CLIENT, AZURE_OPENAI_DEPLOYMENT

    if LLM_PROVIDER == "gemini" and "GEMINI_API_KEY" in os.environ:
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        GEMINI_MODEL = genai.GenerativeModel("models/gemini-2.5-flash")
        USE_GEMINI = True
        print("✅ Gemini API configured.")

    elif LLM_PROVIDER == "openai" and "OPENAI_API_KEY" in os.environ:
        import openai
        openai.api_key = os.environ["OPENAI_API_KEY"]
        USE_OPENAI = True
        print("✅ OpenAI API configured.")

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
        USE_AZURE_OPENAI = True
        print("✅ Azure OpenAI API configured.")
    

    elif LLM_PROVIDER == "claude" and "CLAUDE_API_KEY" in os.environ:
        import anthropic
        CLAUDE_CLIENT = anthropic.Anthropic(api_key=os.environ["CLAUDE_API_KEY"])
        USE_CLAUDE = True
        print("✅ Claude API configured.")

    else:
        print(f"❌ Invalid or missing config for LLM_PROVIDER='{LLM_PROVIDER}'.")

# --- LLM Inference Functions ---
def get_code_topics_gemini(code_snippet, language="java"):
    import google.generativeai as genai
    if not USE_GEMINI:
        return ["AI analysis skipped (Gemini API not configured)"]
    prompt = prompt_mgr.get("code_topic", code_snippet=code_snippet, language=language)
    try:
        response = GEMINI_MODEL.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                response_mime_type="application/json"
            )
        )
        topics_data = json.loads(response.text)
        return topics_data.get("topics", ["N/A (Invalid JSON response format from Gemini)"])
    except Exception as e:
        print(f"❌ Error calling Gemini API: {e}")
        return [f"N/A (API Error: {e})"]

def get_code_topics_openai(code_snippet, language="java"):
    import openai
    if not USE_OPENAI:
        return ["AI analysis skipped (OpenAI API not configured)"]
    prompt = prompt_mgr.get("code_topic", code_snippet=code_snippet, language=language)
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert code analyst. Your response must be a JSON object."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        topics_data = json.loads(response['choices'][0]['message']['content'])
        return topics_data.get("topics", ["N/A (Invalid JSON response format from OpenAI)"])
    except Exception as e:
        print(f"❌ Error calling OpenAI API: {e}")
        return [f"N/A (API Error: {e})"]

def get_code_topics_azure_openai(code_snippet, language="java"):
    import openai
    if not USE_AZURE_OPENAI:
        return ["AI analysis skipped (Azure OpenAI API not configured)"]
    prompt = prompt_mgr.get("code_topic", code_snippet=code_snippet, language=language)
    try:
        response = openai.ChatCompletion.create(
            deployment_id=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are an expert code analyst. Your response must be a JSON object."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        topics_data = json.loads(response['choices'][0]['message']['content'])
        return topics_data.get("topics", ["N/A (Invalid JSON response format from Azure OpenAI)"])
    except Exception as e:
        print(f"❌ Error calling Azure OpenAI API: {e}")
        return [f"N/A (API Error: {e})"]

def get_code_topics_claude(code_snippet, language="java"):
    if not USE_CLAUDE:
        return ["AI analysis skipped (Claude API not configured)"]
    prompt = prompt_mgr.get("code_topic", code_snippet=code_snippet, language=language)
    try:
        response = CLAUDE_CLIENT.messages.create(
            model=CLAUDE_MODEL_NAME,
            max_tokens=1024,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text
        topics_data = json.loads(text)
        return topics_data.get("topics", ["N/A (Invalid JSON response format from Claude)"])
    except Exception as e:
        print(f"❌ Error calling Claude API: {e}")
        return [f"N/A (API Error: {e})"]

# --- Orchestrator Function ---
def analyze_and_store_code_topics(input_json_file_path, output_json_file_path):
    try:
        with open(input_json_file_path, 'r', encoding='utf-8') as f:
            attempts_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading input file: {e}")
        return

    analyzed_data = []
    for i, attempt in enumerate(attempts_data):
        attempt_id = attempt.get('attempt_id', 'N/A')
        code_snippet = attempt.get('code_used_to_generate', '')
        print(f"➡️  Analyzing entry {i+1}/{len(attempts_data)} (Attempt ID: {attempt_id})")

        if code_snippet:
            if USE_GEMINI:
                topics = get_code_topics_gemini(code_snippet)
            elif USE_OPENAI:
                topics = get_code_topics_openai(code_snippet)
            elif USE_AZURE_OPENAI:
                topics = get_code_topics_azure_openai(code_snippet)
            elif USE_CLAUDE:
                topics = get_code_topics_claude(code_snippet)
            else:
                topics = ["No valid LLM provider configured"]
        else:
            topics = ["No code provided"]

        attempt['inferred_code_topics'] = topics
        analyzed_data.append(attempt)

    try:
        os.makedirs(os.path.dirname(output_json_file_path), exist_ok=True)
        with open(output_json_file_path, 'w', encoding='utf-8') as f:
            json.dump(analyzed_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved output to {output_json_file_path}")
    except Exception as e:
        print(f"❌ Error saving output: {e}")

# --- Entry Point ---
if __name__ == "__main__":
    setup_llm()
    analyze_and_store_code_topics(
        'input_data/minimal_data/summary_data.json',
        'json_reports/code_topics_report.json'
    )

# import json
# import os
# from dotenv import load_dotenv
# from prompt_handler import PromptManager
# from openai import OpenAI

# # --- Load environment variables ---
# load_dotenv(dotenv_path='config/.env')
# LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# # --- Flags and Clients ---
# USE_GEMINI = USE_OPENAI = USE_CLAUDE = USE_AZURE_OPENAI = False
# GEMINI_MODEL = None
# OPENAI_MODEL_NAME = "gpt-3.5-turbo"
# CLAUDE_MODEL_NAME = "claude-3-sonnet-20240229"
# CLAUDE_CLIENT = None
# AZURE_OPENAI_DEPLOYMENT = None

# # Azure OpenAI config
# AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")

# # --- Prompt Manager ---
# prompt_mgr = PromptManager()

# # --- LLM Setup Function ---
# def setup_llm():
#     global USE_GEMINI, USE_OPENAI, USE_CLAUDE, USE_AZURE_OPENAI
#     global GEMINI_MODEL, CLAUDE_CLIENT, AZURE_OPENAI_DEPLOYMENT, openai_client

#     if LLM_PROVIDER == "gemini" and "GEMINI_API_KEY" in os.environ:
#         import google.generativeai as genai
#         genai.configure(api_key=os.environ["GEMINI_API_KEY"])
#         GEMINI_MODEL = genai.GenerativeModel("models/gemini-2.5-flash")
#         USE_GEMINI = True
#         print("✅ Gemini API configured.")

#     elif LLM_PROVIDER == "openai" and "OPENAI_API_KEY" in os.environ:
#         import openai
#         openai.api_key = os.environ["OPENAI_API_KEY"]
#         USE_OPENAI = True
#         print("✅ OpenAI API configured.")

#     elif LLM_PROVIDER == "azure" and all(k in os.environ for k in [
#         "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
#         "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_VERSION"
#     ]):
#         AZURE_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
#         USE_AZURE_OPENAI = True
#         openai_client = OpenAI(
#             api_key=AZURE_API_KEY,
#             api_base=AZURE_API_BASE,
#             api_type="azure",
#             api_version=AZURE_API_VERSION,
#         )
#         print("✅ Azure OpenAI API configured.")

#     elif LLM_PROVIDER == "claude" and "CLAUDE_API_KEY" in os.environ:
#         import anthropic
#         CLAUDE_CLIENT = anthropic.Anthropic(api_key=os.environ["CLAUDE_API_KEY"])
#         USE_CLAUDE = True
#         print("✅ Claude API configured.")

#     else:
#         print(f"❌ Invalid or missing config for LLM_PROVIDER='{LLM_PROVIDER}'.")

# # --- LLM Inference Functions ---
# def get_code_topics_gemini(code_snippet, language="java"):
#     import google.generativeai as genai
#     if not USE_GEMINI:
#         return ["AI analysis skipped (Gemini API not configured)"]
#     prompt = prompt_mgr.get("code_topic", code_snippet=code_snippet, language=language)
#     try:
#         response = GEMINI_MODEL.generate_content(
#             prompt,
#             generation_config=genai.types.GenerationConfig(
#                 temperature=0.2,
#                 response_mime_type="application/json"
#             )
#         )
#         topics_data = json.loads(response.text)
#         return topics_data.get("topics", ["N/A (Invalid JSON response format from Gemini)"])
#     except Exception as e:
#         print(f"❌ Error calling Gemini API: {e}")
#         return [f"N/A (API Error: {e})"]

# def get_code_topics_openai(code_snippet, language="java"):
#     import openai
#     if not USE_OPENAI:
#         return ["AI analysis skipped (OpenAI API not configured)"]
#     prompt = prompt_mgr.get("code_topic", code_snippet=code_snippet, language=language)
#     try:
#         response = openai.ChatCompletion.create(
#             model=OPENAI_MODEL_NAME,
#             messages=[
#                 {"role": "system", "content": "You are an expert code analyst. Your response must be a JSON object."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.2
#         )
#         topics_data = json.loads(response['choices'][0]['message']['content'])
#         return topics_data.get("topics", ["N/A (Invalid JSON response format from OpenAI)"])
#     except Exception as e:
#         print(f"❌ Error calling OpenAI API: {e}")
#         return [f"N/A (API Error: {e})"]

# def get_code_topics_azure_openai(code_snippet, language="java"):
#     if not USE_AZURE_OPENAI:
#         return ["AI analysis skipped (Azure OpenAI API not configured)"]
#     prompt = prompt_mgr.get("code_topic", code_snippet=code_snippet, language=language)
#     try:
#         response = openai_client.chat.completions.create(
#             deployment_id=AZURE_OPENAI_DEPLOYMENT,
#             messages=[
#                 {"role": "system", "content": "You are an expert code analyst. Your response must be a JSON object."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.2
#         )
#         text = response.choices[0].message.content
#         topics_data = json.loads(text)
#         return topics_data.get("topics", ["N/A (Invalid JSON response format from Azure OpenAI)"])
#     except Exception as e:
#         print(f"❌ Error calling Azure OpenAI API: {e}")
#         return [f"N/A (API Error: {e})"]

# def get_code_topics_claude(code_snippet, language="java"):
#     if not USE_CLAUDE:
#         return ["AI analysis skipped (Claude API not configured)"]
#     prompt = prompt_mgr.get("code_topic", code_snippet=code_snippet, language=language)
#     try:
#         response = CLAUDE_CLIENT.messages.create(
#             model=CLAUDE_MODEL_NAME,
#             max_tokens=1024,
#             temperature=0.2,
#             messages=[{"role": "user", "content": prompt}]
#         )
#         text = response.content[0].text
#         topics_data = json.loads(text)
#         return topics_data.get("topics", ["N/A (Invalid JSON response format from Claude)"])
#     except Exception as e:
#         print(f"❌ Error calling Claude API: {e}")
#         return [f"N/A (API Error: {e})"]

# # --- Orchestrator Function ---
# def analyze_and_store_code_topics(input_json_file_path, output_json_file_path):
#     try:
#         with open(input_json_file_path, 'r', encoding='utf-8') as f:
#             attempts_data = json.load(f)
#     except Exception as e:
#         print(f"❌ Error loading input file: {e}")
#         return

#     analyzed_data = []
#     for i, attempt in enumerate(attempts_data):
#         attempt_id = attempt.get('attempt_id', 'N/A')
#         code_snippet = attempt.get('code_used_to_generate', '')
#         print(f"➡️  Analyzing entry {i+1}/{len(attempts_data)} (Attempt ID: {attempt_id})")

#         if code_snippet:
#             if USE_GEMINI:
#                 topics = get_code_topics_gemini(code_snippet)
#             elif USE_OPENAI:
#                 topics = get_code_topics_openai(code_snippet)
#             elif USE_AZURE_OPENAI:
#                 topics = get_code_topics_azure_openai(code_snippet)
#             elif USE_CLAUDE:
#                 topics = get_code_topics_claude(code_snippet)
#             else:
#                 topics = ["No valid LLM provider configured"]
#         else:
#             topics = ["No code provided"]

#         attempt['inferred_code_topics'] = topics
#         analyzed_data.append(attempt)

#     try:
#         os.makedirs(os.path.dirname(output_json_file_path), exist_ok=True)
#         with open(output_json_file_path, 'w', encoding='utf-8') as f:
#             json.dump(analyzed_data, f, indent=2, ensure_ascii=False)
#         print(f"✅ Saved output to {output_json_file_path}")
#     except Exception as e:
#         print(f"❌ Error saving output: {e}")

# # --- Entry Point ---
# if __name__ == "__main__":
#     setup_llm()
#     analyze_and_store_code_topics(
#         'input_data/minimal_data/summary_data.json',
#         'json_reports/code_topics_report.json'
#     )
