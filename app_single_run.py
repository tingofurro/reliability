import os
import ujson as json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils_experiments import get_experiment_type
import re

st.set_page_config(page_title="Single Run Viewer", layout="wide")

def ansi_to_html(text):
    ansi_colors = {
        '30': 'black', '31': 'red', '32': 'green', '33': 'yellow',
        '34': 'blue', '35': 'magenta', '36': 'cyan', '37': 'white',
        '90': 'gray', '91': '#ff6b6b', '92': '#51cf66', '93': '#ffd43b',
        '94': '#4dabf7', '95': '#cc5de8', '96': '#22b8cf', '97': 'white',
        '0': 'reset'
    }
    
    html_parts = []
    open_span = False
    
    parts = re.split(r'\x1b\[(\d+)m', text)
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part:
                part = part.replace('<', '&lt;').replace('>', '&gt;')
                html_parts.append(part)
        else:
            if open_span:
                html_parts.append('</span>')
                open_span = False
            
            if part == '0':
                continue
            elif part in ansi_colors and part != '0':
                color = ansi_colors[part]
                html_parts.append(f'<span style="color: {color};">')
                open_span = True
    
    if open_span:
        html_parts.append('</span>')
    
    return ''.join(html_parts)

def load_run_data(exp_name):
    exp_path = f"experiments/{exp_name}"
    
    if not os.path.exists(f"{exp_path}/args.json"):
        return None, "args.json not found"
    if not os.path.exists(f"{exp_path}/logs.jsonl"):
        return None, "logs.jsonl not found"
    
    with open(f"{exp_path}/args.json", "r") as f:
        exp_args = json.load(f)
    
    logs = []
    with open(f"{exp_path}/logs.jsonl", "r") as f:
        for line in f:
            logs.append(json.loads(line))
    
    if len(logs) == 0:
        return None, "No logs found"
    
    return {"args": exp_args, "logs": logs}, None

def extract_time_series(logs):
    iterations = [log["iteration"] for log in logs]
    mean_train_score = [log.get("mean_train_score", 0) for log in logs]
    mean_eval_score = [log.get("mean_eval_score", 0) for log in logs]
    unique_answers = [log.get("unique_answers", 0) for log in logs]
    num_eval_responses = [log.get("num_eval_responses", 0) for log in logs]
    uniqueness = [log.get("uniqueness", 0) for log in logs]
    num_unique_correct_answers = [log.get("num_unique_correct_answers", 0) for log in logs]
    
    correct_logprobs = [np.mean(log.get("correct_logprobs", [0])) if log.get("correct_logprobs") else 0 for log in logs]
    incorrect_logprobs = [np.mean(log.get("incorrect_logprobs", [0])) if log.get("incorrect_logprobs") else 0 for log in logs]
    
    backprop_errors = [1 if log.get("backprop_error") is not None else 0 for log in logs]
    
    return {
        "iterations": iterations,
        "mean_train_score": mean_train_score,
        "mean_eval_score": mean_eval_score,
        "unique_answers": unique_answers,
        "num_eval_responses": num_eval_responses,
        "uniqueness": uniqueness,
        "correct_logprobs": correct_logprobs,
        "incorrect_logprobs": incorrect_logprobs,
        "num_unique_correct_answers": num_unique_correct_answers,
        "backprop_errors": backprop_errors
    }

st.title("Single Run Viewer")

experiments = sorted([exp for exp in os.listdir("experiments") if os.path.isdir(f"experiments/{exp}")], reverse=True)

if len(experiments) == 0:
    st.error("No experiments found in the experiments folder")
    st.stop()

st.sidebar.header("Select Run")
selected_exp = st.sidebar.selectbox("Experiment", experiments, index=0)

if selected_exp:
    run_data, error = load_run_data(selected_exp)
    
    if error:
        st.error(f"Error loading run: {error}")
        st.stop()
    
    args = run_data["args"]
    logs = run_data["logs"]
    
    st.subheader(f"Run: {selected_exp}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Task ID", args.get("task_id", "N/A"))
    col2.metric("Experiment Type", get_experiment_type(args))
    col3.metric("Learning Rate", args.get("learning_rate", "N/A"))
    col4.metric("Total Iterations", len(logs))
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Group Size", args.get("group_size", "N/A"))
    col2.metric("Tree Degree", args.get("tree_degree", "N/A"))
    col3.metric("Tree Depth", args.get("tree_depth", "N/A"))
    
    tab1, tab2 = st.tabs(["Metrics", "Train Logs"])
    
    with tab1:
        time_series = extract_time_series(logs)
        
        fig = make_subplots(
            rows=5, cols=2,
            subplot_titles=(
                "Mean Train Score", "Mean Eval Score",
                "Unique Answers", "Num Eval Responses",
                "Uniqueness (%)", "Num Unique Correct Answers",
                "Correct Log Probs", "Incorrect Log Probs",
                "Backprop Errors", ""
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.12
        )
        
        iterations = time_series["iterations"]
        
        fig.add_trace(go.Scatter(x=iterations, y=time_series["mean_train_score"], mode='lines+markers', name='Mean Train Score', line=dict(color='#1f77b4'), showlegend=False), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=iterations, y=time_series["mean_eval_score"], mode='lines+markers', name='Mean Eval Score', line=dict(color='#ff7f0e'), showlegend=False), row=1, col=2)
        
        fig.add_trace(go.Scatter(x=iterations, y=time_series["unique_answers"], mode='lines+markers', name='Unique Answers', line=dict(color='#2ca02c'), showlegend=False), row=2, col=1)
        
        fig.add_trace(go.Scatter(x=iterations, y=time_series["num_eval_responses"], mode='lines+markers', name='Num Eval Responses', line=dict(color='#d62728'), showlegend=False), row=2, col=2)
        
        fig.add_trace(go.Scatter(x=iterations, y=time_series["uniqueness"], mode='lines+markers', name='Uniqueness', line=dict(color='#9467bd'), showlegend=False), row=3, col=1)
        
        fig.add_trace(go.Scatter(x=iterations, y=time_series["num_unique_correct_answers"], mode='lines+markers', name='Num Unique Correct', line=dict(color='#8c564b'), showlegend=False), row=3, col=2)
        
        fig.add_trace(go.Scatter(x=iterations, y=time_series["correct_logprobs"], mode='lines+markers', name='Correct Log Probs', line=dict(color='#e377c2'), showlegend=False), row=4, col=1)
        
        fig.add_trace(go.Scatter(x=iterations, y=time_series["incorrect_logprobs"], mode='lines+markers', name='Incorrect Log Probs', line=dict(color='#7f7f7f'), showlegend=False), row=4, col=2)
        
        fig.add_trace(go.Scatter(x=iterations, y=time_series["backprop_errors"], mode='lines+markers', name='Backprop Errors', line=dict(color='#bcbd22'), fill='tozeroy', showlegend=False), row=5, col=1)
        
        for i in range(1, 6):
            for j in range(1, 3):
                fig.update_xaxes(title_text="Iteration", row=i, col=j)
        
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        fig.update_yaxes(title_text="Percentage", row=3, col=1)
        fig.update_yaxes(title_text="Count", row=3, col=2)
        fig.update_yaxes(title_text="Log Prob", row=4, col=1)
        fig.update_yaxes(title_text="Log Prob", row=4, col=2)
        fig.update_yaxes(title_text="Error (0/1)", row=5, col=1)
        
        fig.update_layout(height=1600, hovermode='x unified')
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Detailed Logs")
        
        with st.expander("View raw log data"):
            for i, log in enumerate(logs):
                st.json(log)
        
        st.subheader("Experiment Arguments")
        with st.expander("View experiment arguments"):
            st.json(args)
    
    with tab2:
        run_logs_path = f"experiments/{selected_exp}/run_logs.ans"
        if os.path.exists(run_logs_path):
            with open(run_logs_path, "r") as f:
                run_logs_content = f.read()
            
            html_content = ansi_to_html(run_logs_content)
            st.markdown(f'<div style="background-color: #0e1117; padding: 20px; border-radius: 5px; height: 800px; overflow-y: scroll; font-family: monospace; white-space: pre-wrap; font-size: 14px;">{html_content}</div>', unsafe_allow_html=True)
        else:
            st.warning("run_logs.ans file not found for this experiment")

