import os
import ujson as json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict

st.set_page_config(page_title="Training Analysis", layout="wide")

def load_experiment_data():
    experiments = sorted([exp for exp in os.listdir("experiments")])
    necessary_files = ["args.json", "logs.jsonl", "unique_answers.jsonl"]
    
    exp_results = []
    for exp in experiments:
        if any(not os.path.exists(f"experiments/{exp}/{f}") for f in necessary_files):
            continue
        
        with open(f"experiments/{exp}/args.json", "r") as f:
            exp_args = json.load(f)
        
        learning_rate = exp_args["learning_rate"]

        experiment_type = ""
        sample_strategy = exp_args.get("sample_strategy", "iid")
        if sample_strategy == "tree":
            experiment_type = f"tree-{exp_args['tree_depth']}-{exp_args['tree_degree']}"
        else:
            experiment_type = f"iid-{exp_args['group_size']}"
        
        exp_logs = []
        with open(f"experiments/{exp}/logs.jsonl", "r") as f:
            for line in f:
                exp_logs.append(json.loads(line))
        
        task_id = exp_args["task_id"]
        
        with open(f"experiments/{exp}/unique_answers.jsonl", "r") as f:
            unique_answers = [json.loads(line) for line in f]
        
        already_seen_answers = set()
        num_new_answers = []
        for answers in unique_answers:
            this_answers = set(answers)
            new_answers = this_answers - already_seen_answers
            num_new_answers.append(len(new_answers))
            already_seen_answers |= this_answers
        
        mean_eval_score = [log["mean_eval_score"] for log in exp_logs]
        uniqueness = [log["uniqueness"]/100.0 for log in exp_logs]
        
        correct_logprobs = [np.mean(log["correct_logprobs"]) for log in exp_logs]
        incorrect_logprobs = [np.mean(log["incorrect_logprobs"]) for log in exp_logs]
        num_unique_correct_answers = [log["num_unique_correct_answers"] for log in exp_logs]
        
        is_success = mean_eval_score[-1] >= 0.99
        
        exp_results.append({
            "task_id": task_id,
            "mean_eval_score": mean_eval_score,
            "uniqueness": uniqueness,
            "learning_rate": learning_rate,
            "experiment_type": experiment_type,
            "correct_logprobs": correct_logprobs,
            "incorrect_logprobs": incorrect_logprobs,
            "num_unique_correct_answers": num_unique_correct_answers,
            "num_new_answers": num_new_answers,
            "is_success": is_success
        })
    
    return exp_results

# Load data
try:
    exp_results = load_experiment_data()
    st.sidebar.success(f"Loaded {len(exp_results)} experiments")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Get unique task_ids and experiment_types
task_ids = sorted(list(set([res["task_id"] for res in exp_results])))
experiment_types = sorted(list(set([res["experiment_type"] for res in exp_results])))

# Sidebar controls
st.sidebar.header("Filters")
selected_task_id = st.sidebar.selectbox("Task ID", ["all"] + task_ids, index=0)
show_only_success = st.sidebar.checkbox("Show only successful runs", value=False)
max_iter = st.sidebar.slider("Max Iterations", min_value=5, max_value=50, value=25)

# Title
st.title("Training Analysis Dashboard")
st.markdown(f"**Task:** {selected_task_id} | **Filter:** {'Success only' if show_only_success else 'All runs'}")

# Process data for plots
def process_data_for_plots(task_id, success_only, max_iterations):
    all_mean_eval_scores, all_uniquenesses = {}, {}
    run_counts = {}
    
    for experiment_type in experiment_types:
        all_mean_eval_scores[experiment_type], all_uniquenesses[experiment_type] = [], []
        this_results = [res for res in exp_results if (res["task_id"] == task_id or task_id == "all") and res["experiment_type"] == experiment_type]
        
        if success_only:
            this_results = [res for res in this_results if res["is_success"]]
        
        run_counts[experiment_type] = len(this_results)
        
        for res in this_results:
            mean_eval_score = res["mean_eval_score"][:max_iterations]
            if len(mean_eval_score) < max_iterations:
                mean_eval_score += [mean_eval_score[-1]] * (max_iterations - len(mean_eval_score))
            
            uniqueness = res["uniqueness"][:max_iterations]
            if len(uniqueness) < max_iterations:
                uniqueness += [uniqueness[-1]] * (max_iterations - len(uniqueness))
            
            all_mean_eval_scores[experiment_type].append(mean_eval_score)
            all_uniquenesses[experiment_type].append(uniqueness)
    
    return all_mean_eval_scores, all_uniquenesses, run_counts

# Process data for "All runs" and "Success only"
all_data = process_data_for_plots(selected_task_id, False, max_iter)
success_data = process_data_for_plots(selected_task_id, True, max_iter)

# Create 2x2 subplot
fig = make_subplots(rows=2, cols=2, subplot_titles=("Mean Eval Scores - All", "Uniqueness - All", "Mean Eval Scores - Success", "Uniqueness - Success"), vertical_spacing=0.12, horizontal_spacing=0.1)

# Colors for experiment types
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# Plot "All runs" - Mean Eval Scores (row 1, col 1)
for idx, experiment_type in enumerate(experiment_types):
    if len(all_data[0][experiment_type]) > 0:
        mean_scores = np.mean(all_data[0][experiment_type], axis=0)
        std_scores = np.std(all_data[0][experiment_type], axis=0)
        iterations = list(range(len(mean_scores)))
        
        fig.add_trace(go.Scatter(x=iterations, y=mean_scores, mode='lines', name=f"{experiment_type}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{experiment_type}", showlegend=True, hovertemplate=f'{experiment_type}<br>Iteration: %{{x}}<br>Score: %{{y:.3f}}<br>Runs: {all_data[2][experiment_type]}<extra></extra>'), row=1, col=1)

# Plot "All runs" - Uniqueness (row 1, col 2)
for idx, experiment_type in enumerate(experiment_types):
    if len(all_data[1][experiment_type]) > 0:
        mean_uniqueness = np.mean(all_data[1][experiment_type], axis=0)
        iterations = list(range(len(mean_uniqueness)))
        
        fig.add_trace(go.Scatter(x=iterations, y=mean_uniqueness, mode='lines', name=f"{experiment_type}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{experiment_type}", showlegend=False, hovertemplate=f'{experiment_type}<br>Iteration: %{{x}}<br>Uniqueness: %{{y:.3f}}<br>Runs: {all_data[2][experiment_type]}<extra></extra>'), row=1, col=2)

# Plot "Success only" - Mean Eval Scores (row 2, col 1)
for idx, experiment_type in enumerate(experiment_types):
    if len(success_data[0][experiment_type]) > 0:
        mean_scores = np.mean(success_data[0][experiment_type], axis=0)
        iterations = list(range(len(mean_scores)))
        
        fig.add_trace(go.Scatter(x=iterations, y=mean_scores, mode='lines', name=f"{experiment_type}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{experiment_type}", showlegend=False, hovertemplate=f'{experiment_type}<br>Iteration: %{{x}}<br>Score: %{{y:.3f}}<br>Runs: {success_data[2][experiment_type]}<extra></extra>'), row=2, col=1)

# Plot "Success only" - Uniqueness (row 2, col 2)
for idx, experiment_type in enumerate(experiment_types):
    if len(success_data[1][experiment_type]) > 0:
        mean_uniqueness = np.mean(success_data[1][experiment_type], axis=0)
        iterations = list(range(len(mean_uniqueness)))
        
        fig.add_trace(go.Scatter(x=iterations, y=mean_uniqueness, mode='lines', name=f"{experiment_type}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{experiment_type}", showlegend=False, hovertemplate=f'{experiment_type}<br>Iteration: %{{x}}<br>Uniqueness: %{{y:.3f}}<br>Runs: {success_data[2][experiment_type]}<extra></extra>'), row=2, col=2)

# Update axes labels
fig.update_xaxes(title_text="Iteration", row=1, col=1)
fig.update_xaxes(title_text="Iteration", row=1, col=2)
fig.update_xaxes(title_text="Iteration", row=2, col=1)
fig.update_xaxes(title_text="Iteration", row=2, col=2)

fig.update_yaxes(title_text="Mean Eval Score", row=1, col=1)
fig.update_yaxes(title_text="Uniqueness", row=1, col=2)
fig.update_yaxes(title_text="Mean Eval Score", row=2, col=1)
fig.update_yaxes(title_text="Uniqueness", row=2, col=2)

# Update layout
fig.update_layout(height=800, hovermode='closest', legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02))

st.plotly_chart(fig, use_container_width=True)

# Display run counts
st.subheader("Run Counts by Condition")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**All Runs:**")
    for experiment_type in experiment_types:
        count = all_data[2][experiment_type]
        st.markdown(f"- {experiment_type}: **{count}** runs")

with col2:
    st.markdown("**Success Only:**")
    for experiment_type in experiment_types:
        count = success_data[2][experiment_type]
        st.markdown(f"- {experiment_type}: **{count}** runs")

# Summary statistics
st.subheader("Summary Statistics")
total_runs = len([res for res in exp_results if (res["task_id"] == selected_task_id or selected_task_id == "all")])
success_runs = len([res for res in exp_results if (res["task_id"] == selected_task_id or selected_task_id == "all") and res["is_success"]])
success_rate = (success_runs / total_runs * 100) if total_runs > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Runs", total_runs)
col2.metric("Successful Runs", success_runs)
col3.metric("Success Rate", f"{success_rate:.1f}%")

