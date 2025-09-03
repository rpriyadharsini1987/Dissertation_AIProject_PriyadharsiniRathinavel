
#  Testing Framework

A unified framework for analyzing, evaluating, and visualizing student code attempts, LLM-generated questions/answers, and compliance metrics for educational hackathons and coding assessments.

---

## 📦 Project Structure

```
├── answer_comparison.py                # Compare LLM and student answers
├── data_extractor.py                   # Extract and preprocess main dataset
├── extract_formatted_data.py           # Format and extract data for analysis
├── prompt_handler.py                   # Prompt management for LLMs
├── topic_question_student_flag.py      # Analyze student question relevance flags
├── topics.py                           # Extract code topics from submissions
├── compliance/
│   └── compliance.py                   # Compliance checks for answers/questions
├── config/
│   └── .env                            # Environment variables (LLM provider, etc.)
├── deepeval/
│   └── data_formatter.py               # Data formatting for DeepEval metrics
├── frontend_bot/
│   └── bot_ui.py                       # (Experimental) Frontend bot UI
├── input_data/
│   ├── bulk_data/                      # Raw input datasets
│   └── minimal_data/                   # Processed/filtered datasets
├── json_reports/                       # Output reports (JSON)
├── not_required/
│   └── extract.py                      # (Deprecated) Extraction scripts
├── prompts/                            # Prompt templates for LLMs
├── ragas/                              # Ragas metric evaluation scripts
├── streamlit/
│   └── streamlit_combined_dashboard.py # Streamlit dashboard for visualization
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

---

## 🚀 Quickstart

### 1. Install Python 3.11

```sh
brew install python@3.11
```

### 2. Create and Activate Virtual Environment

```sh
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```sh
pip install -r requirements.txt
pip install openai
python -m spacy download en_core_web_sm
pip install --upgrade langchain_openai
```

### 4. Prepare Data

- **Extract and preprocess data:**
```sh
python3 data_extractor.py
```
- **Extract code topics:**
```sh
python3 topics.py
```
- **Analyze student question flags:**
```sh
python3 topic_question_student_flag.py
```
- **Compare LLM and student answers:**
```sh
python3 answer_comparison.py
```
- **Compliance checks:**
```sh
python3 compliance/compliance.py
```
- **Evaluate Ragas metrics:**
```sh
python3 ragas/evaluate_ragas.py
```
- **Validate Ragas metrics:**
```sh
python3 ragas/validate_ragas_metrics.py
```

### 5. Run the Dashboard

```sh
streamlit run streamlit/streamlit_combined_dashboard.py
```

---

## 📊 Features

- **Data Extraction & Formatting:** Scripts to clean, filter, and format student code and question/answer data.
- **Topic Extraction:** Automatic inference of code topics from student submissions.
- **LLM & Student Answer Comparison:** Analyze and visualize how student answers compare to LLM-generated answers.
- **Compliance Checking:** Ensure answers/questions meet compliance requirements.
- **Metric Evaluation:** Ragas and custom metrics for answer/question quality.
- **Interactive Dashboard:** Streamlit-based dashboard for unified visualization and analysis.

---

## ⚙️ Configuration

- Set your LLM provider and other environment variables in `config/.env`:
  ```
  LLM_PROVIDER=gemini  # or openai, etc.
  ```

---

## 📚 Requirements

See [`requirements.txt`](requirements.txt) for all dependencies, including:
- `streamlit`
- `python-dotenv`
- `spacy`
- `langchain_openai`
- `ragas`
- `deepeval`
- and others

---

## 📁 Data

- Place your input datasets in `input_data/bulk_data/` or `input_data/minimal_data/`.
- Output reports are generated in `json_reports/`.

---

## 📝 Notes

- Some scripts and folders (e.g., `not_required/`) are deprecated or experimental.
- Prompts for LLMs are stored in the `prompts/` directory.
- The dashboard supports multiple LLM providers (set in `.env`).

---

## 🖥️ Dashboard Overview

The Streamlit dashboard provides three main tabs:
1. **Code Topics Report:** Review code topics and generated questions for each student attempt.
2. **Switch Model Report:** Analyze question relevance and flagging accuracy.
3. **LLM vs Student Answer Match:** Compare LLM and student answers, with verdicts for each question.

---

## 📄 License

MIT License (add your license here if different).

---


