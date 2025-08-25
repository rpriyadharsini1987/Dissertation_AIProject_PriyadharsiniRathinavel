
import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import os
import inspect, sys

def render_eval_csv_generator(df):
    """Show a button that generates + downloads eval_scores.csv (demo metrics)."""
    import re, numpy as np, pandas as pd, streamlit as st

    st.markdown("**⬇️ Download the eval_scores.csv here**")
    if st.button("Generate & download eval_scores.csv", key="gen_eval_scores", use_container_width=True):
        base = df.groupby("QuestionID").agg(
            flag_rate=("MarkedAsWrong", "mean"),
            q=("Question", "first"),
            ans=("CorrectAnswer", "first"),
            code=("CodeSnippet", "first"),
        ).reset_index()

        def _tokens(s: str):
            s = str(s or "").lower()
            return set(re.findall(r"[a-zA-Z_]\w+", s))

        def _jaccard(a: str, b: str):
            A, B = _tokens(a), _tokens(b)
            if not A and not B:
                return 0.0
            return len(A & B) / max(1, len(A | B))

        base["ragas_relevancy"] = base.apply(lambda r: _jaccard(r["q"], r["ans"]), axis=1)
        base["ragas_faithfulness"] = base.apply(lambda r: _jaccard(r["ans"], r["code"]), axis=1)
        base["ragas_similarity"] = (1.0 - base["flag_rate"].fillna(0)).clip(0, 1)
        base["ragas_correctness"] = (0.6*base["ragas_relevancy"] + 0.4*(1 - base["flag_rate"].fillna(0))).clip(0, 1)
        base["deepeval_hallucination"] = (1 - base["ragas_faithfulness"]).clip(0, 1) * 0.4
        base["deepeval_factual_consistency"] = base["ragas_faithfulness"].clip(0, 1)

        out_cols = [
            "QuestionID", "ragas_relevancy", "ragas_faithfulness", "ragas_similarity",
            "ragas_correctness", "deepeval_hallucination", "deepeval_factual_consistency"
        ]
        out_df = base[out_cols].round(3)

        st.download_button(
            "⬇️ Download the eval_scores.csv here",
            out_df.to_csv(index=False).encode("utf-8"),
            file_name="eval_scores.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.session_state["eval_scores_df"] = out_df
        st.success(f"Generated {len(out_df)} rows. Using these scores until you upload a new file.")


# ===============================
# App Config
# ===============================
st.set_page_config(page_title="Code QA Dashboard (Enhanced)", layout="wide")
st.title("📊 Code-Derived Question QA Dashboard – Enhanced")
st.caption("Upload your exported JSON to explore accuracy, taxonomy, item analysis, LLM-based item validity, and actionable insights.")

# Offer a one-click way to download this app's source
try:
    _src = inspect.getsource(sys.modules[__name__])
    st.download_button("💾 Download this app (.py)", _src, file_name="AutoMCQ_Dashboard_Integrated.py", mime="text/x-python")
except Exception:
    pass

# ===============================
# File Uploader
# ===============================
uploaded_file = st.file_uploader("📁 Upload your dataset.json", type=["json"])

def download_df(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False)
    st.download_button(f"⬇️ {label}", data=csv, file_name=filename, mime="text/csv")

# ---------- Bloom helpers ----------
BLOOM_KEYWORDS = {
    "remember": r"\b(define|list|name|state|identify|recall|what is|which|when|where)\b",
    "understand": r"\b(explain|summarize|describe|classify|interpret|why|give reason)\b",
    "apply": r"\b(apply|use|compute|calculate|implement|execute|solve|run)\b",
    "analyze": r"\b(analy[sz]e|debug|trace|compare|differentiate|distinguish|break down|cause)\b",
    "evaluate": r"\b(evaluate|justify|assess|argue|critique|verify|prove|decide)\b",
    "create": r"\b(create|design|compose|build|develop|formulate|write (a )?program)\b",
}
BLOOM_SYNONYMS = {
    "analyse": "analyze",
    "analysis": "analyze",
    "remembering": "remember",
    "understanding": "understand",
    "application": "apply",
    "applying": "apply",
    "evaluation": "evaluate",
    "creating": "create",
}
BLOOM_TO_LABELS = {
    "remember": ("Remember", "Factual"),
    "understand": ("Understand", "Conceptual"),
    "apply": ("Apply", "Procedural"),
    "analyze": ("Analyze", "Conceptual"),
    "evaluate": ("Evaluate", "Metacognitive"),
    "create": ("Create", "Metacognitive"),
}
bloom_choices = ["remember","understand","apply","analyze","evaluate","create"]

with st.sidebar.expander("⚙️ Bloom Rules", expanded=False):
    st.caption("Choose how to classify frequent patterns. These override keyword rules.")
    custom_levels = {
        "output_prediction": st.selectbox("Output / print / result questions", bloom_choices, index=bloom_choices.index("apply")),
        "time_complexity": st.selectbox("Time complexity / Big-O", bloom_choices, index=bloom_choices.index("analyze")),
        "debug_trace": st.selectbox("Debug / trace / bug finding", bloom_choices, index=bloom_choices.index("analyze")),
        "syntax_validity": st.selectbox("Syntax validity / declaration correctness", bloom_choices, index=bloom_choices.index("understand")),
        "concept_mcq": st.selectbox("Concept MCQs (which/true about ...)", bloom_choices, index=bloom_choices.index("understand")),
        "implement_write": st.selectbox("Implement / use API / compute", bloom_choices, index=bloom_choices.index("apply")),
        "design_build": st.selectbox("Design / build / create", bloom_choices, index=bloom_choices.index("create")),
        "justify_prove": st.selectbox("Justify / verify / prove / evaluate", bloom_choices, index=bloom_choices.index("evaluate")),
    }

def pick_bloom_from_text(qtext: str) -> str:
    qt = " " + (qtext or "").lower() + " "
    if re.search(r"\b(output|print(ed)?|result)\b", qt): return custom_levels["output_prediction"]
    if re.search(r"\b(time complexity|big-?o|o\()\b", qt): return custom_levels["time_complexity"]
    if re.search(r"\b(debug|trace|bug|error|fix)\b", qt): return custom_levels["debug_trace"]
    if re.search(r"\b(valid|syntax|declaration)\b", qt): return custom_levels["syntax_validity"]
    if re.search(r"\b(which of the following|true about)\b", qt): return custom_levels["concept_mcq"]
    if re.search(r"\b(implement|use|compute|calculate|run)\b", qt): return custom_levels["implement_write"]
    if re.search(r"\b(design|build|create|develop|write (a )?program)\b", qt): return custom_levels["design_build"]
    if re.search(r"\b(justify|assess|argue|verify|prove|evaluate)\b", qt): return custom_levels["justify_prove"]
    for level, pat in BLOOM_KEYWORDS.items():
        if re.search(pat, qt, flags=re.I):
            return level
    return "unknown"

def normalize_bloom(raw_val: str, qtext: str) -> str:
    s = ("" if raw_val is None else str(raw_val)).strip().lower()
    if s in BLOOM_SYNONYMS: s = BLOOM_SYNONYMS[s]
    if s in BLOOM_TO_LABELS: return s
    s2 = re.sub(r"[^a-z]", "", s)
    s2 = BLOOM_SYNONYMS.get(s2, s2)
    if s2 in BLOOM_TO_LABELS: return s2
    return pick_bloom_from_text(qtext)

def nice_cognitive_label(key: str) -> str:
    return BLOOM_TO_LABELS.get(key, ("Unknown","Unknown"))[0]

def nice_knowledge_label(key: str) -> str:
    return BLOOM_TO_LABELS.get(key, ("Unknown","Unknown"))[1]

# ===============================
# Main
# ===============================
if uploaded_file:
    # ---------- Parse JSON safely ----------
    try:
        raw_data = json.load(uploaded_file)
        data = raw_data.get("data", []) if isinstance(raw_data, dict) else raw_data
        if not isinstance(data, list):
            st.error("Invalid structure: 'data' must be a list.")
            st.stop()

        rows, skipped = [], []

        def get_bloom_any(q: dict):
            for k in ["BloomLevel","bloomLevel","bloom_level","bloom","taxonomy","bloomTaxonomy"]:
                if k in q and q[k] not in (None, ""):
                    return q[k]
            return None

        for idx, attempt in enumerate(data):
            if not isinstance(attempt, dict):
                skipped.append({"Index": idx, "Reason": "Non-dict attempt"})
                continue

            questions = attempt.get("questions")
            if not isinstance(questions, list):
                skipped.append({"Index": idx, "Reason": "'questions' not a list"})
                continue

            for q in questions:
                if not isinstance(q, dict):
                    continue
                try:
                    question_text = q.get("Question")
                    bloom_raw = get_bloom_any(q)

                    rows.append({
                        "AttemptID": attempt.get("ID", ""),
                        "StudentEmail": attempt.get("studentUsername", ""),
                        "Timestamp": attempt.get("Timestamp", ""),
                        "CodeSnippet": attempt.get("CodeUsedToGenerate", ""),
                        "Topics": attempt.get("questionGenerationProperties", {}).get("question_topics", ""),
                        "QuestionID": q.get("ID"),
                        "Question": question_text,
                        "StudentAnswer": (q.get("StudentAnswer", {}) or {}).get("optionText", ""),
                        "CorrectAnswer": (q.get("Answer", {}) or {}).get("optionText", ""),
                        "Correct": q.get("Correct", False),
                        "MarkedAsWrong": q.get("this_doesnt_seem_right", False),
                        "BloomLevelRaw": bloom_raw,
                        "BloomNormKey": normalize_bloom(bloom_raw, question_text),
                    })
                except Exception as e:
                    skipped.append({"Index": idx, "Reason": str(e)})

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    if not rows:
        st.error("No valid questions found.")
        st.stop()

    df = pd.DataFrame(rows)

    # ---------- Clean & enrich ----------
    student_map = {email: f"Student {i+1}" for i, email in enumerate(df["StudentEmail"].dropna().unique())}
    df["Student"] = df["StudentEmail"].map(student_map)
    df.drop(columns=["StudentEmail"], inplace=True)

    df["Correct"] = df["Correct"].astype(bool)
    df["MarkedAsWrong"] = df["MarkedAsWrong"].astype(bool)

    def _topics_to_list(val):
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
        if pd.isna(val):
            return []
        return [x.strip() for x in str(val).split(",") if x.strip()]

    df["Topics"] = df["Topics"].apply(_topics_to_list)

    TOPIC_MAP = {
        r"\bfor\s*\(|while\s*\(|for\s+\w+\s+in\b": "loops",
        r"\barray|list|vector|ArrayList|List\b": "arrays-lists",
        r"\bmap|dict|hash|HashMap\b": "hash-maps",
        r"\bclass|object|inherit|polymorph|interface\b": "oop",
        r"\btry\s*:|catch|except|finally|Exception\b": "exceptions",
        r"\brecursive|recursion\b": "recursion",
        r"\bcomplexity|O\(|Big\-O\b": "complexity",
        r"\bfile\.|open\(|read\(|write\(|Scanner\b": "file-io",
        r"\bthread|concurrent|async|await|synchronized\b": "concurrency",
        r"\bsql|select|insert|update|delete|join\b": "sql",
    }
    def normalize_topics(topic_list, code_text):
        topics = set(t.lower() for t in (topic_list or []) if t)
        code = str(code_text or "")
        for pattern, tag in TOPIC_MAP.items():
            if re.search(pattern, code, flags=re.I):
                topics.add(tag)
        return sorted(topics) if topics else ["uncategorized"]

    df["Topics"] = df.apply(lambda r: normalize_topics(r["Topics"], r["CodeSnippet"]), axis=1)

    df["CognitiveProcess"] = df["BloomNormKey"].apply(lambda k: BLOOM_TO_LABELS.get(k, ("Unknown","Unknown"))[0])
    df["KnowledgeDimension"] = df["BloomNormKey"].apply(lambda k: BLOOM_TO_LABELS.get(k, ("Unknown","Unknown"))[1])

    def guess_qtype(q: str) -> str:
        ql = str(q or "").lower()
        if "what will be printed" in ql or "output" in ql: return "Output-Prediction"
        if "time complexity" in ql or "big-o" in ql: return "Complexity"
        if "find the bug" in ql or "debug" in ql: return "Debugging"
        if "valid" in ql and ("syntax" in ql or "declaration" in ql): return "Syntax"
        if "which of the following" in ql or "true about" in ql: return "Concept"
        return "Other"

    df["QuestionType"] = df["Question"].apply(guess_qtype)
    df["Score"] = df["Correct"].astype(int)

    with st.expander("📋 Data Preview (first 50 rows)", expanded=False):
        st.dataframe(df.head(50))
        download_df(df, "cleaned_data.csv", "Download All Data")
    # ---------- Filters ----------
    st.sidebar.header("🎯 Filters")
    students = sorted(df["Student"].dropna().unique().tolist())
    topics_all = sorted({t for lst in df["Topics"] for t in lst if t})
    bloom_levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create", "Unknown"]
    qtypes = sorted(df["QuestionType"].unique())

    selected_students = st.sidebar.multiselect("Students", students)
    selected_topics = st.sidebar.multiselect("Topics", topics_all)
    selected_bloom = st.sidebar.multiselect("Cognitive Process", bloom_levels)
    selected_qtypes = st.sidebar.multiselect("Question Type", qtypes)

    filtered = df.copy()
    if selected_students:
        filtered = filtered[filtered["Student"].isin(selected_students)]
    if selected_topics:
        filtered = filtered[filtered["Topics"].apply(lambda lst: any(t in lst for t in selected_topics))]
    if selected_bloom:
        filtered = filtered[filtered["CognitiveProcess"].isin(selected_bloom)]
    if selected_qtypes:
        filtered = filtered[filtered["QuestionType"].isin(selected_qtypes)]

    # ==============================================================
    # LLM Evaluation Fusion Layer (loads eval scores & quality gate)
    # ==============================================================
    with st.sidebar.expander("🧪 Item Quality (LLM Eval)", expanded=False):
        st.caption(
            "Upload eval_scores from your Azure/GitLab pipeline. The app will also look for "
            "eval_scores.parquet/csv/json in the working folder. Expected columns include: "
            "QuestionID, ragas_relevancy, ragas_faithfulness, ragas_similarity, ragas_correctness, "
            "deepeval_hallucination, deepeval_factual_consistency (names are auto-normalised)."
        )
        eval_upload = st.file_uploader("Upload eval_scores (csv/json/parquet)", type=["csv","json","parquet"], key="eval_upload")

        st.markdown("**Quality Gate thresholds** (tune for your cohort):")
        relevancy_thr     = st.slider("Relevancy ≥",         0.50, 0.95, 0.75, 0.01)
        faithfulness_thr  = st.slider("Faithfulness ≥",      0.50, 0.95, 0.75, 0.01)
        halluc_thr        = st.slider("Max Hallucination ≤", 0.00, 1.00, 0.20, 0.01)
        min_attempts_thr  = st.number_input("Min attempts for gate", min_value=1, max_value=50, value=8, step=1)
        disc_thr          = st.slider("Min discrimination ≥", -1.0, 1.0, 0.10, 0.01)
        flag_thr          = st.slider("Max flag rate ≤",      0.0, 1.0, 0.25, 0.01)

        st.markdown("---")
        only_pass_box  = st.checkbox("Show only items passing the Quality Gate", value=False)
        min_ivi_slider = st.slider("Minimum Item Validity Index (IVI)", 0.0, 1.0, 0.0, 0.05)

        render_eval_csv_generator(df)

    def _load_eval_local_or_uploaded(upload):
        if upload is not None:
            name = upload.name.lower()
            try:
                if name.endswith(".csv"):
                    return pd.read_csv(upload)
                elif name.endswith(".json"):
                    return pd.read_json(upload)
                elif name.endswith(".parquet"):
                    try:
                        return pd.read_parquet(upload)
                    except Exception as e:
                        st.warning(f"Could not read parquet: {e}")
                        return None
            except Exception as e:
                st.warning(f"Failed to read uploaded eval file: {e}")
                return None
        for p in ["eval_scores.parquet", "eval_scores.csv", "eval_scores.json"]:
            if os.path.exists(p):
                try:
                    if p.endswith(".parquet"):
                        return pd.read_parquet(p)
                    elif p.endswith(".csv"):
                        return pd.read_csv(p)
                    else:
                        return pd.read_json(p)
                except Exception as e:
                    st.warning(f"Found {p} but failed to read: {e}")
        return None

    eval_df = _load_eval_local_or_uploaded(eval_upload)
    if eval_df is None and "eval_scores_df" in st.session_state:
        eval_df = st.session_state["eval_scores_df"]
        st.info("Using the eval scores you just generated in this session.")

    per_item = None

    if eval_df is not None:
        # Normalise columns
        rename_map = {
            "QuestionId": "QuestionID",
            "question_id": "QuestionID",
            "AnswerRelevancy": "ragas_relevancy",
            "Relevancy": "ragas_relevancy",
            "Faithfulness": "ragas_faithfulness",
            "AnswerCorrectness": "ragas_correctness",
            "AnswerSimilarity": "ragas_similarity",
            "Hallucination": "deepeval_hallucination",
            "FactualConsistency": "deepeval_factual_consistency",
        }
        eval_df = eval_df.rename(columns=rename_map)
        needed = [
            "QuestionID","ragas_relevancy","ragas_faithfulness","ragas_similarity",
            "ragas_correctness","deepeval_hallucination","deepeval_factual_consistency"
        ]
        for c in needed:
            if c not in eval_df.columns:
                eval_df[c] = np.nan
        eval_df = eval_df[needed]

        # ---------------- NEW: robust ID normalization ----------------
        def _norm_id_series(s: pd.Series) -> pd.Series:
            # 1) Coerce to string
            out = s.astype(str).str.strip()
            # 2) Fix Excel-like "123.0"
            out = out.str.replace(r"\.0$", "", regex=True)
            # 3) Remove accidental '.00', spaces, etc.
            out = out.str.replace(r"\.00$", "", regex=True)
            # 4) Treat 'nan'/'None' as empty
            out = out.replace({"nan": "", "None": "", "NaT": ""})
            return out

        df["QuestionID"] = _norm_id_series(df["QuestionID"])
        eval_df["QuestionID"] = _norm_id_series(eval_df["QuestionID"])

        # Drop duplicates after normalization
        eval_df = eval_df.drop_duplicates(subset=["QuestionID"], keep="last")

        # Merge into attempt-level df
        df = df.merge(eval_df, on="QuestionID", how="left")

        # Per-item aggregates (attempt-space)
        per_item = df.groupby("QuestionID").agg(
            Attempts=("Score","count"),
            Corrects=("Score","sum"),
            FlagRate=("MarkedAsWrong","mean"),
            ragas_relevancy=("ragas_relevancy","mean"),
            ragas_faithfulness=("ragas_faithfulness","mean"),
            ragas_similarity=("ragas_similarity","mean"),
            ragas_correctness=("ragas_correctness","mean"),
            hallucination=("deepeval_hallucination","mean"),
            fact_consistency=("deepeval_factual_consistency","mean")
        ).reset_index()
        per_item["Difficulty"] = per_item["Corrects"] / per_item["Attempts"]

        totals = df.groupby("Student")["Score"].sum().rename("TotalScore")
        tmp_disc = df.merge(totals, on="Student", how="left")

        def _pb(group: pd.DataFrame) -> float:
            if group["Score"].nunique() < 2 or group["TotalScore"].var(ddof=1) == 0:
                return np.nan
            return group[["Score","TotalScore"]].corr().iloc[0,1]

        disc = tmp_disc.groupby("QuestionID").apply(_pb).rename("Discrimination").reset_index()
        per_item = per_item.merge(disc, on="QuestionID", how="left")

        def clip01(x):
            try:
                return float(min(max(x, 0.0), 1.0))
            except Exception:
                return np.nan

        for col in ["ragas_relevancy","ragas_faithfulness","ragas_similarity","hallucination"]:
            per_item[col] = per_item[col].apply(lambda x: clip01(x) if pd.notna(x) else x)

        per_item["IVI"] = (
            0.4 * per_item["ragas_relevancy"].fillna(0) +
            0.4 * per_item["ragas_faithfulness"].fillna(0) +
            0.2 * per_item["ragas_similarity"].fillna(0) -
            0.2 * per_item["hallucination"].fillna(0)
        ).clip(0,1)

        per_item["MinAttemptsOK"] = per_item["Attempts"] >= int(min_attempts_thr)
        per_item["DiscOK"]        = per_item["Discrimination"].fillna(-1) >= float(disc_thr)
        per_item["FlagsOK"]       = per_item["FlagRate"].fillna(0) <= float(flag_thr)
        per_item["EvalOK"]        = (
            (per_item["ragas_relevancy"].fillna(0)    >= float(relevancy_thr)) &
            (per_item["ragas_faithfulness"].fillna(0) >= float(faithfulness_thr)) &
            (per_item["hallucination"].fillna(0)      <= float(halluc_thr))
        )
        per_item["QualityGate"] = per_item["MinAttemptsOK"] & per_item["DiscOK"] & per_item["FlagsOK"] & per_item["EvalOK"]

        def reasons(r):
            out = []
            if not r["MinAttemptsOK"]: out.append("few attempts")
            if not r["DiscOK"]: out.append("low discrimination")
            if not r["FlagsOK"]: out.append("high flag rate")
            if not r["EvalOK"]: out.append("low eval (relevancy/faithfulness/hallucination)")
            return ", ".join(out) if out else ""
        per_item["ReviewReason"] = per_item.apply(reasons, axis=1)

        df = df.merge(per_item[["QuestionID","IVI","QualityGate","ReviewReason"]], on="QuestionID", how="left")

        # Apply Quality Gate controls
        filtered = df.copy()
        if selected_students:
            filtered = filtered[filtered["Student"].isin(selected_students)]
        if selected_topics:
            filtered = filtered[filtered["Topics"].apply(lambda lst: any(t in lst for t in selected_topics))]
        if selected_bloom:
            filtered = filtered[filtered["CognitiveProcess"].isin(selected_bloom)]
        if selected_qtypes:
            filtered = filtered[filtered["QuestionType"].isin(selected_qtypes)]
        if only_pass_box:
            filtered = filtered[filtered["QualityGate"] == True]
        if min_ivi_slider > 0:
            filtered = filtered[filtered["IVI"].fillna(0) >= min_ivi_slider]
    else:
        st.info("No evaluation scores found (eval_scores.*). You can still use the dashboard; validity filters are disabled.")

    # ===============================
    # Accuracy
    # ===============================
    st.subheader("✅ Overall Accuracy")
    acc_counts = filtered["Correct"].value_counts(dropna=False)
    labels = ["Correct" if idx is True else "Incorrect" for idx in acc_counts.index]
    fig_pie = px.pie(values=acc_counts.values, names=labels, title="Answer Accuracy Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)

    # ===============================
    # "This Doesn't Seem Right" Flag Analysis
    # ===============================

    st.subheader("🚩 'This Doesn't Seem Right' Flag Counts")
    flag_counts = filtered["MarkedAsWrong"].value_counts(dropna=False)
    flag_labels = ["Flagged as Wrong" if val else "Not Flagged" for val in flag_counts.index]
    # Build a small DataFrame so Plotly Express has unambiguous columns
    flag_df = pd.DataFrame({"FlagStatus": flag_labels, "Count": flag_counts.values})
    fig_flag = px.bar(
        flag_df,
        x="FlagStatus",
        y="Count",
        text="Count",
        color="FlagStatus",
        title="'This Doesn't Seem Right' – True/False Count"
    )
    fig_flag.update_traces(textposition="outside")
    st.plotly_chart(fig_flag, use_container_width=True)

    # Download counts as CSV
    download_df(flag_df, "flag_true_false_counts.csv", "Download Flag Counts")

    st.markdown("**📈 Flag Trend Over Time**")
    tmp = filtered.copy()
    tmp["TimestampDT"] = pd.to_datetime(tmp["Timestamp"], errors="coerce", utc=True)
    tmp["Date"] = tmp["TimestampDT"].dt.date
    tmp = tmp.dropna(subset=["Date"])

    tab_overall, tab_students = st.tabs(["Overall", "By Student"])

    with tab_overall:
        if not tmp.empty:
            daily = tmp.groupby(["Date", "MarkedAsWrong"]).size().reset_index(name="Count")
            pivot = daily.pivot(index="Date", columns="MarkedAsWrong", values="Count").fillna(0).rename(
                columns={False: "Not Flagged", True: "Flagged"}
            ).sort_index()

            daily_long = pivot.reset_index().melt(id_vars="Date", value_vars=["Not Flagged", "Flagged"],
                                                  var_name="FlagStatus", value_name="Count")
            fig_daily = px.area(
                daily_long, x="Date", y="Count", color="FlagStatus",
                title="Daily Flag Counts (Overall, stacked)",
            )
            st.plotly_chart(fig_daily, use_container_width=True)
            download_df(daily_long, "daily_flag_counts_overall.csv", "Download Daily Flag Counts (Overall)")

            total = (pivot["Flagged"] + pivot["Not Flagged"]).replace(0, np.nan)
            rate_out = (pivot["Flagged"] / total * 100).round(2).dropna().reset_index()
            rate_out.columns = ["Date", "Flag %"]
            if not rate_out.empty:
                fig_rate = px.line(rate_out, x="Date", y="Flag %", title="Daily % Flagged (Overall)")
                st.plotly_chart(fig_rate, use_container_width=True)
                download_df(rate_out, "daily_flag_rate_overall.csv", "Download Daily Flag % (Overall)")
            else:
                st.info("Not enough dated rows to compute daily flag percentage.")
        else:
            st.info("No valid timestamps to show overall trends. (Check the 'Timestamp' field in your JSON.)")

    with tab_students:
        if not tmp.empty:
            ds = tmp.groupby(["Date", "Student", "MarkedAsWrong"]).size().reset_index(name="Count")
            ds_pivot = ds.pivot_table(index=["Date", "Student"], columns="MarkedAsWrong", values="Count", aggfunc="sum").fillna(0)
            if True not in ds_pivot.columns: ds_pivot[True] = 0
            if False not in ds_pivot.columns: ds_pivot[False] = 0
            ds_pivot = ds_pivot.rename(columns={False: "Not Flagged", True: "Flagged"}).reset_index().sort_values(["Student","Date"])

            ds_pivot["Total"] = ds_pivot["Flagged"] + ds_pivot["Not Flagged"]
            ds_pivot["Flag %"] = np.where(ds_pivot["Total"] > 0, (ds_pivot["Flagged"] / ds_pivot["Total"] * 100).round(2), np.nan)

            all_students = sorted(ds_pivot["Student"].dropna().unique().tolist())
            sel_students = st.multiselect("Students to plot", all_students, default=all_students[:5])

            if sel_students:
                ds_sel = ds_pivot[ds_pivot["Student"].isin(sel_students)].copy()

                fig_cnt = px.line(
                    ds_sel, x="Date", y="Flagged", color="Student",
                    title="Daily Flagged Counts by Student",
                )
                st.plotly_chart(fig_cnt, use_container_width=True)

                ds_rate = ds_sel.dropna(subset=["Flag %"])
                if not ds_rate.empty:
                    fig_rate = px.line(
                        ds_rate, x="Date", y="Flag %", color="Student",
                        title="Daily % Flagged by Student",
                    )
                    st.plotly_chart(fig_rate, use_container_width=True)
                else:
                    st.info("No days with calculable % for selected students.")

                download_df(ds_sel[["Date","Student","Flagged","Not Flagged","Total"]], "daily_flag_counts_by_student.csv", "Download Daily Flag Counts (By Student)")
                download_df(ds_sel[["Date","Student","Flag %"]].dropna(), "daily_flag_rate_by_student.csv", "Download Daily Flag % (By Student)")
            else:
                st.info("Select at least one student to display.")
        else:
            st.info("No valid timestamps to show student-wise trends. (Check the 'Timestamp' field in your JSON.)")

    # ===============================
    # Student-Topic Weakness (Bar)
    # ===============================
    st.subheader("📉 Student-Topic Weakness (Incorrect Only)")
    incorrect_df = filtered[filtered["Score"] == 0].explode("Topics")
    bar_data = incorrect_df.groupby(["Student", "Topics"]).size().reset_index(name="IncorrectCount")
    if not bar_data.empty:
        fig_bar = px.bar(
            bar_data, x="Student", y="IncorrectCount", color="Topics", barmode="group",
            title="# Incorrect by Student & Topic", labels={"IncorrectCount": "# Incorrect"}, height=500
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        download_df(bar_data, "student_topic_weakness.csv", "Bar Chart Data")
    else:
        st.info("No incorrect answers available for visualization.")

    # ===============================
    # Bloom’s Taxonomy – % Correct (normalized)
    # ===============================
    st.subheader("📚 Bloom’s Taxonomy Analysis")
    bloom_summary = filtered.groupby(["CognitiveProcess", "Correct"]).size().unstack(fill_value=0)
    bloom_summary["Total"] = bloom_summary.sum(axis=1)
    bloom_summary["% Correct"] = (bloom_summary.get(True, 0) / bloom_summary["Total"] * 100).round(2)
    fig_bloom = px.bar(
        bloom_summary.reset_index(),
        x="CognitiveProcess", y="% Correct",
        title="% Correct by Bloom Level (normalized)"
    )
    st.plotly_chart(fig_bloom, use_container_width=True)
    download_df(bloom_summary.reset_index(), "bloom_summary.csv", "Download Bloom Summary")

    # ===============================
    # Item Analysis (Difficulty, Discrimination, KR-20)
    # ===============================
    st.subheader("🧪 Item Analysis (Difficulty, Discrimination, KR-20)")

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        min_attempts = st.number_input(
            "Min attempts per item", min_value=1, max_value=50, value=3, step=1,
            help="Items with fewer attempts are excluded from stats/plot."
        )
    with colB:
        discr_method = st.selectbox(
            "Discrimination method", ["Point-biserial", "Upper-Lower 27%"], index=0,
            help="Use point-biserial where possible; UL27 is more robust on small N."
        )
    with colC:
        ycap = st.selectbox("Y-axis range (Discrimination)", ["auto", "-0.5 to 0.5", "-1 to 1"], index=1)

    item_stats = filtered.groupby("QuestionID").agg(
        Attempts=("Score", "count"),
        Corrects=("Score", "sum"),
        FlaggedRate=("MarkedAsWrong", "mean"),
    ).reset_index()

    if not item_stats.empty:
        item_stats["Difficulty"] = item_stats["Corrects"] / item_stats["Attempts"]
        item_stats = item_stats[item_stats["Attempts"] >= int(min_attempts)].copy()

        totals = filtered.groupby("Student")["Score"].sum().rename("TotalScore")
        tmp = filtered.merge(totals, on="Student", how="left")

        if discr_method == "Point-biserial":
            def point_biserial(group: pd.DataFrame) -> float:
                if group["Score"].nunique() < 2: return np.nan
                if group["TotalScore"].var(ddof=1) == 0: return np.nan
                return group[["Score", "TotalScore"]].corr().iloc[0, 1]
            disc = tmp.groupby("QuestionID").apply(point_biserial).rename("Discrimination").reset_index()
        else:
            totals_all = filtered.groupby("Student")["Score"].sum()
            if len(totals_all) >= 8:
                n27 = max(1, int(round(len(totals_all) * 0.27)))
                ordered = totals_all.sort_values()
                lower = set(ordered.index[:n27])
                upper = set(ordered.index[-n27:])
                def ul27(group: pd.DataFrame) -> float:
                    g = group.copy()
                    pU = g[g["Student"].isin(upper)]["Score"].mean() if any(g["Student"].isin(upper)) else np.nan
                    pL = g[g["Student"].isin(lower)]["Score"].mean() if any(g["Student"].isin(lower)) else np.nan
                    return pU - pL if pd.notna(pU) and pd.notna(pL) else np.nan
                disc = tmp.groupby("QuestionID").apply(ul27).rename("Discrimination").reset_index()
            else:
                disc = pd.DataFrame({"QuestionID": item_stats["QuestionID"], "Discrimination": np.nan})

        item_stats = item_stats.merge(disc, on="QuestionID", how="left")

        k = filtered["QuestionID"].nunique()
        if k >= 2:
            p = item_stats["Difficulty"].clip(0, 1)
            q = 1 - p
            var_total = filtered.groupby("Student")["Score"].sum().var(ddof=1)
            kr20 = (k / (k - 1)) * (1 - (p.mul(q).sum() / var_total)) if var_total and not np.isnan(var_total) else None
        else:
            kr20 = None

        cols_show = ["QuestionID", "Attempts", "Corrects", "Difficulty", "Discrimination", "FlaggedRate"]

        # Join IVI if available
        if per_item is not None:
            cols_show += ["IVI"]
            item_stats = item_stats.merge(per_item[["QuestionID","IVI"]], on="QuestionID", how="left")

        st.metric("KR-20 Reliability", f"{kr20:.3f}" if kr20 is not None and not np.isnan(kr20) else "N/A")
        st.dataframe(item_stats[cols_show].sort_values(["Discrimination", "Difficulty"], ascending=[True, True]))
        download_df(item_stats, "item_stats.csv", "Download Item Stats")

        plot_df = item_stats.dropna(subset=["Difficulty", "Discrimination"]).copy()
        st.markdown("**🎯 Question Quality (Difficulty vs Discrimination)**")
        if not plot_df.empty:
            fig_scatter = px.scatter(
                plot_df, x="Difficulty", y="Discrimination",
                hover_data=["QuestionID", "Attempts"],
                title="Item Difficulty vs Discrimination"
            )
            if ycap == "-0.5 to 0.5":
                fig_scatter.update_yaxes(range=[-0.5, 0.5])
            elif ycap == "-1 to 1":
                fig_scatter.update_yaxes(range=[-1, 1])
            fig_scatter.update_xaxes(range=[-0.05, 1.05])
            fig_scatter.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No items met the criteria for plotting (increase ‘Min attempts’ or switch to UL-27).")

        if per_item is not None:
            with st.expander("🧯 Item Review Queue (fails any rule)", expanded=False):
                review = per_item[per_item["QualityGate"] == False].copy()
                review = review.sort_values(["IVI","Discrimination","Attempts"], ascending=[True, True, True])
                st.dataframe(review[
                    ["QuestionID","Attempts","Difficulty","Discrimination","FlagRate",
                     "ragas_relevancy","ragas_faithfulness","hallucination","IVI","ReviewReason"]
                ], use_container_width=True)
                download_df(review, "item_review_queue.csv", "Download Review Queue")
    else:
        st.info("Not enough data for item analysis.")

    # ===============================
    # Donut Charts – Clarity Views
    # ===============================
    st.subheader("🍩 Donut Views")

    tab_topic_overall, tab_topic_students = st.tabs(["Topic Donut – Overall/Single", "Topic Donut – By Student"])
    with tab_topic_overall:
        scope_students = ["(All)"] + sorted(filtered["Student"].dropna().unique().tolist())
        sel_stu = st.selectbox("Student (for topic donut)", scope_students, index=0, key="donut_student_topic")
        inc = filtered[filtered["Score"] == 0].explode("Topics")
        if sel_stu != "(All)":
            inc = inc[inc["Student"] == sel_stu]
        donut_topic = inc["Topics"].value_counts().reset_index()
        donut_topic.columns = ["Topic", "Incorrect"]
        if not donut_topic.empty:
            fig_donut_topic = px.pie(
                donut_topic, names="Topic", values="Incorrect", hole=0.5,
                title=(f"Incorrect by Topic – {'All Students' if sel_stu=='(All)' else sel_stu}")
            )
            fig_donut_topic.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_donut_topic, use_container_width=True)
            download_df(donut_topic, "donut_incorrect_by_topic.csv", "Download Topic Donut Data")
        else:
            st.info("No incorrect answers for the selected scope.")

    with tab_topic_students:
        all_stu = sorted(filtered["Student"].dropna().unique().tolist())
        sel_multi = st.multiselect("Students to compare", all_stu, default=all_stu[:4], key="topic_multi_sel")
        if sel_multi:
            cols = 2
            rows = int(np.ceil(len(sel_multi) / cols))
            fig = make_subplots(rows=rows, cols=cols, specs=[[{"type": "domain"}]*cols for _ in range(rows)],
                                subplot_titles=sel_multi)
            r = c = 1
            for s in sel_multi:
                inc_s = filtered[(filtered["Score"] == 0) & (filtered["Student"] == s)].explode("Topics")
                vc = inc_s["Topics"].value_counts()
                if vc.empty:
                    fig.add_trace(go.Pie(labels=["(none)"], values=[1], hole=0.5, showlegend=False), row=r, col=c)
                else:
                    fig.add_trace(go.Pie(labels=vc.index, values=vc.values, hole=0.5, showlegend=False), row=r, col=c)
                c += 1
                if c > cols:
                    c = 1
                    r += 1
            fig.update_layout(title_text="Incorrect by Topic — Side-by-side by Student", height=350*rows, margin=dict(t=80))
            st.plotly_chart(fig, use_container_width=True)

            out = []
            for s in sel_multi:
                inc_s = filtered[(filtered["Score"] == 0) & (filtered["Student"] == s)].explode("Topics")
                vc = inc_s["Topics"].value_counts().reset_index()
                vc.columns = ["Topic","Incorrect"]
                vc.insert(0, "Student", s)
                out.append(vc)
            out_df = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["Student","Topic","Incorrect"])
            download_df(out_df, "topic_donut_by_student.csv", "Download Topic Donut Data (By Student)")
        else:
            st.info("Select at least one student to compare.")

    tab_bloom_overall, tab_bloom_students = st.tabs(["Bloom Donut – Overall", "Bloom Donut – By Student"])
    with tab_bloom_overall:
        blo_inc = filtered[filtered["Score"] == 0].groupby("CognitiveProcess").size().reset_index(name="Incorrect")
        if not blo_inc.empty:
            fig_donut_bloom = px.pie(
                blo_inc, names="CognitiveProcess", values="Incorrect", hole=0.5,
                title="Incorrect Share by Bloom’s Cognitive Process"
            )
            fig_donut_bloom.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_donut_bloom, use_container_width=True)
            download_df(blo_inc, "donut_incorrect_by_bloom.csv", "Download Bloom Donut Data")
        else:
            st.info("No incorrect answers to analyze by Bloom level.")

    with tab_bloom_students:
        all_stu2 = sorted(filtered["Student"].dropna().unique().tolist())
        sel_multi2 = st.multiselect("Students to compare", all_stu2, default=all_stu2[:4], key="bloom_multi_sel")
        if sel_multi2:
            cols_b = 2
            rows_b = int(np.ceil(len(sel_multi2) / cols_b))
            fig2 = make_subplots(rows=rows_b, cols=cols_b, specs=[[{"type": "domain"}]*cols_b for _ in range(rows_b)],
                                 subplot_titles=sel_multi2)
            r = c = 1
            for s in sel_multi2:
                inc_s = filtered[(filtered["Score"] == 0) & (filtered["Student"] == s)]
                vc = inc_s.groupby("CognitiveProcess").size().sort_values(ascending=False) if not inc_s.empty else pd.Series(dtype=int)
                if vc.empty:
                    fig2.add_trace(go.Pie(labels=["(none)"], values=[1], hole=0.5, showlegend=False), row=r, col=c)
                else:
                    fig2.add_trace(go.Pie(labels=vc.index, values=vc.values, hole=0.5, showlegend=False), row=r, col=c)
                c += 1
                if c > cols_b:
                    c = 1
                    r += 1
            fig2.update_layout(title_text="Incorrect by Bloom — Side-by-side by Student", height=350*rows_b, margin=dict(t=80))
            st.plotly_chart(fig2, use_container_width=True)

            out_b = []
            for s in sel_multi2:
                inc_s = filtered[(filtered["Score"] == 0) & (filtered["Student"] == s)]
                vc = inc_s.groupby("CognitiveProcess").size().reset_index(name="Incorrect")
                vc.insert(0, "Student", s)
                out_b.append(vc)
            out_b_df = pd.concat(out_b, ignore_index=True) if out_b else pd.DataFrame(columns=["Student","CognitiveProcess","Incorrect"])
            download_df(out_b_df, "bloom_donut_by_student.csv", "Download Bloom Donut Data (By Student)")
        else:
            st.info("Select at least one student to compare.")

    # ===============================
    # Tables & Drilldown
    # ===============================
    st.subheader("📊 Student × Topic – Incorrect Rate & Attempts")
    _heat_src = filtered.explode("Topics")
    if not _heat_src.empty:
        pair = (
            _heat_src.groupby(["Student", "Topics"], dropna=False)["Score"]
            .agg(IncorrectRate=lambda s: 1 - s.mean(), Attempts="count")
            .reset_index()
        )
        if not pair.empty:
            pair_disp = pair.copy()
            pair_disp["Incorrect %"] = (pair_disp["IncorrectRate"] * 100).round(0).astype(int).astype(str) + "%"
            pair_disp = pair_disp[["Student", "Topics", "Incorrect %", "Attempts"]]

            with st.expander("View full table / export", expanded=False):
                st.dataframe(pair_disp.sort_values(["Student", "Incorrect %"], ascending=[True, False]), use_container_width=True)
                download_df(pair.sort_values(["Student", "IncorrectRate"], ascending=[True, False]),
                            "student_topic_table.csv", "Download Student×Topic Table (raw)")

            n = st.number_input("Show worst N topics per student", min_value=1, max_value=10, value=3, step=1)
            worstN = (pair.sort_values(["Student", "IncorrectRate"], ascending=[True, False])
                            .groupby("Student").head(int(n)))
            worstN_disp = worstN.copy()
            worstN_disp["Incorrect %"] = (worstN_disp["IncorrectRate"] * 100).round(0).astype(int).astype(str) + "%"
            worstN_disp = worstN_disp[["Student", "Topics", "Incorrect %", "Attempts"]]
            st.dataframe(worstN_disp, use_container_width=True)
            download_df(worstN, "worst_topics_per_student.csv", f"Download Worst {int(n)} per Student (raw)")
    else:
        st.info("No topic data to build tables.")

    st.subheader("🔎 Drilldown: Missed Questions")
    _students = ["(All)"] + sorted(filtered["Student"].dropna().unique().tolist())
    _topics = ["(All)"] + sorted({t for lst in filtered["Topics"] for t in lst if t})
    sel_student = st.selectbox("Student", _students, index=0)
    sel_topic = st.selectbox("Topic", _topics, index=0)
    detail = filtered.copy()
    if sel_student != "(All)":
        detail = detail[detail["Student"] == sel_student]
    if sel_topic != "(All)":
        detail = detail[detail["Topics"].apply(lambda lst: sel_topic in lst)]
    detail = detail[detail["Score"] == 0]

    if not detail.empty:
        # Build a clean, de-duplicated column list
        base_cols = ["QuestionID", "Student", "Topics", "Question", "StudentAnswer", "CorrectAnswer", "MarkedAsWrong", "Timestamp"]
        cols = base_cols.copy()
        if "IVI" in detail.columns:
            cols = ["QuestionID", "IVI"] + base_cols  # ensure QuestionID first, IVI second when present

        # De-duplicate while preserving order and keep only existing columns
        seen = set()
        cols = [c for c in cols if (c in detail.columns and (c not in seen and not seen.add(c)))]
        st.dataframe(detail[cols], use_container_width=True)
        download_df(detail[cols], "drilldown_incorrect.csv", "Download Drilldown Rows")
    else:
        st.info("No incorrect rows for the selected filters.")

    # ===============================
    # Heatmap (end)

    # ===============================
    # Heatmap (end)
    # ===============================
    st.subheader("🔥 Weakness Heatmap (Incorrect Rate)")
    heat = filtered.explode("Topics")
    if not heat.empty:
        pair = (
            heat.groupby(["Student", "Topics"], dropna=False)["Score"]
                .agg(IncorrectRate=lambda s: 1 - s.mean(), Attempts="count")
                .reset_index()
        )
        if not pair.empty:
            pivot = pair.pivot(index="Student", columns="Topics", values="IncorrectRate")
            counts = pair.pivot(index="Student", columns="Topics", values="Attempts").fillna(0).astype(int)

            row_order = pivot.mean(axis=1).sort_values(ascending=False).index if pivot.shape[0] else pivot.index
            col_order = pivot.mean(axis=0).sort_values(ascending=False).index if pivot.shape[1] else pivot.columns
            pivot = pivot.loc[row_order, col_order]
            counts = counts.loc[row_order, col_order]

            height = int(max(450, 28 * max(1, len(pivot.index)) + 140))
            cell_count = pivot.shape[0] * pivot.shape[1]
            show_text = cell_count > 0 and cell_count <= 200
            text = None
            if show_text:
                perc = (pivot.fillna(0).values * 100).round(0).astype(int)
                cts = counts.reindex_like(pivot).fillna(0).astype(int).values
                text = np.where(
                    cts == 0, "",
                    (perc.astype(str) + "%" + " (" + cts.astype(str) + ")")
                )

            fig_heat = px.imshow(
                pivot,
                aspect="auto",
                origin="lower",
                color_continuous_scale="RdYlGn_r",
                zmin=0, zmax=1,
                labels=dict(x="Topic", y="Student", color="Incorrect Rate"),
                text_auto=text if show_text else False,
            )
            fig_heat.update_traces(
                customdata=counts.reindex_like(pivot).values,
                hovertemplate=(
                    "Student: %{y}<br>Topic: %{x}<br>Incorrect Rate: %{z:.0%}<br>Attempts: %{customdata}<extra></extra>"
                ),
            )
            fig_heat.update_layout(
                height=height,
                coloraxis_colorbar=dict(title="Incorrect Rate", tickformat=".0%"),
                margin=dict(l=10, r=10, t=60, b=10),
                title={"text": "Incorrect Rate by Student × Topic", "x": 0.5},
            )
            fig_heat.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_heat, use_container_width=True)
            download_df(pivot.reset_index(), "student_topic_heatmap.csv", "Heatmap Data")
        else:
            st.info("Not enough data to plot heatmap.")
    else:
        st.info("No topic data to plot heatmap.")
else:
    st.info("👆 Upload a JSON file to begin.")