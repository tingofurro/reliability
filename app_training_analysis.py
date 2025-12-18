import os
import ujson as json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
from utils_experiments import get_experiment_type

st.set_page_config(page_title="Training Analysis", layout="wide")

SAVED_FILTERS_FILE = "saved_filters.json"

def load_saved_filters():
    if os.path.exists(SAVED_FILTERS_FILE):
        with open(SAVED_FILTERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_filters_to_file(filters_dict):
    with open(SAVED_FILTERS_FILE, "w") as f:
        json.dump(filters_dict, f, indent=2)

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
        experiment_type = get_experiment_type(exp_args)

        git_version = exp_args.get("git_version", None)

        sample_strategy = exp_args.get("sample_strategy", None)
        tree_degree = exp_args.get("tree_degree", None)
        tree_depth = exp_args.get("tree_depth", None)
        group_size = exp_args.get("group_size", None)
        
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
            "is_success": is_success,
            "sample_strategy": sample_strategy,
            "tree_degree": tree_degree,
            "tree_depth": tree_depth,
            "group_size": group_size
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

# Get unique tree parameters
tree_degrees = sorted(list(set([res["tree_degree"] for res in exp_results if res["tree_degree"] is not None])))
tree_depths = sorted(list(set([res["tree_depth"] for res in exp_results if res["tree_depth"] is not None])))
group_sizes = sorted(list(set([res["group_size"] for res in exp_results if res["group_size"] is not None])))

# Initialize session state for saved filters
if "saved_filters" not in st.session_state:
    st.session_state.saved_filters = load_saved_filters()
if "filter_to_load" not in st.session_state:
    st.session_state.filter_to_load = None

# Sidebar controls
st.sidebar.header("Filters")
selected_task_id = st.sidebar.selectbox("Task ID", ["all"] + task_ids, index=0)
max_iter = st.sidebar.slider("Max Iterations", min_value=5, max_value=50, value=25)

st.sidebar.subheader("Run Filter (JSON)")

# Saved filters management
saved_filter_names = list(st.session_state.saved_filters.keys())
if saved_filter_names:
    st.sidebar.markdown("**Load Saved Filter:**")
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        selected_saved_filter = st.selectbox("Saved Filters", [""] + saved_filter_names, label_visibility="collapsed")
    with col2:
        if st.button("Load"):
            if selected_saved_filter:
                st.session_state.filter_to_load = st.session_state.saved_filters[selected_saved_filter]
                st.rerun()
    
    if selected_saved_filter:
        if st.sidebar.button(f"Delete '{selected_saved_filter}'"):
            del st.session_state.saved_filters[selected_saved_filter]
            save_filters_to_file(st.session_state.saved_filters)
            st.rerun()

# Determine default value for text area
default_filter_text = "{}"
if st.session_state.filter_to_load is not None:
    default_filter_text = json.dumps(st.session_state.filter_to_load, indent=2)
    st.session_state.filter_to_load = None

run_filter_text = st.sidebar.text_area("Filter by args (e.g., {\"tree_degree\": 2, \"tree_depth\": 13})", value=default_filter_text, height=100, key="run_filter_input")

# Parse run filter
try:
    run_filter = json.loads(run_filter_text)
    if not isinstance(run_filter, dict):
        st.sidebar.error("Filter must be a JSON object")
        run_filter = {}
except json.JSONDecodeError as e:
    st.sidebar.error(f"Invalid JSON: {e}")
    run_filter = {}

# Save current filter
if run_filter:
    st.sidebar.markdown("**Save Current Filter:**")
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        new_filter_name = st.text_input("Filter name", label_visibility="collapsed", placeholder="Enter filter name")
    with col2:
        if st.button("Save"):
            if new_filter_name:
                st.session_state.saved_filters[new_filter_name] = run_filter
                save_filters_to_file(st.session_state.saved_filters)
                st.sidebar.success(f"Saved '{new_filter_name}'")
                st.rerun()
            else:
                st.sidebar.error("Enter a name")

# Function to check if experiment matches filter
def matches_filter(exp_result, filter_dict):
    if not filter_dict:
        return True
    for key, value in filter_dict.items():
        if exp_result.get(key) != value:
            return False
    return True

# Title
st.title("Training Analysis Dashboard")
filter_info = f"**Task:** {selected_task_id}"
if run_filter:
    filter_info += f" | **Run Filter:** {json.dumps(run_filter)}"
st.markdown(filter_info)

# Process data for plots
def process_data_for_plots(task_id, success_only, max_iterations, run_filter_dict):
    all_mean_eval_scores, all_uniquenesses, all_num_correct_answers = {}, {}, {}
    all_correct_logprobs, all_incorrect_logprobs = {}, {}
    run_counts = {}
    
    for experiment_type in experiment_types:
        all_mean_eval_scores[experiment_type], all_uniquenesses[experiment_type] = [], []
        all_num_correct_answers[experiment_type] = []
        all_correct_logprobs[experiment_type], all_incorrect_logprobs[experiment_type] = [], []
        this_results = [res for res in exp_results if (res["task_id"] == task_id or task_id == "all") and res["experiment_type"] == experiment_type]
        
        # Apply run filter
        this_results = [res for res in this_results if matches_filter(res, run_filter_dict)]
        
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
            
            num_correct_answers = res["num_unique_correct_answers"][:max_iterations]
            if len(num_correct_answers) < max_iterations:
                num_correct_answers += [num_correct_answers[-1]] * (max_iterations - len(num_correct_answers))
            
            correct_logprobs = res["correct_logprobs"][:max_iterations]
            if len(correct_logprobs) < max_iterations:
                correct_logprobs += [correct_logprobs[-1]] * (max_iterations - len(correct_logprobs))
            
            incorrect_logprobs = res["incorrect_logprobs"][:max_iterations]
            if len(incorrect_logprobs) < max_iterations:
                incorrect_logprobs += [incorrect_logprobs[-1]] * (max_iterations - len(incorrect_logprobs))
            
            all_mean_eval_scores[experiment_type].append(mean_eval_score)
            all_uniquenesses[experiment_type].append(uniqueness)
            all_num_correct_answers[experiment_type].append(num_correct_answers)
            all_correct_logprobs[experiment_type].append(correct_logprobs)
            all_incorrect_logprobs[experiment_type].append(incorrect_logprobs)
    
    return all_mean_eval_scores, all_uniquenesses, all_num_correct_answers, all_correct_logprobs, all_incorrect_logprobs, run_counts

# Process data for all runs
all_data = process_data_for_plots(selected_task_id, False, max_iter, run_filter)

# Create 2x3 subplot
fig = make_subplots(rows=2, cols=3, subplot_titles=("Mean Eval Scores", "Uniqueness", "Num Unique Correct Answers", "Correct Logprobs", "Incorrect Logprobs", ""), vertical_spacing=0.15, horizontal_spacing=0.08)

# Colors for experiment types
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# Plot Mean Eval Scores (row 1, col 1)
for idx, experiment_type in enumerate(experiment_types):
    if len(all_data[0][experiment_type]) > 0:
        mean_scores = np.mean(all_data[0][experiment_type], axis=0)
        iterations = list(range(len(mean_scores)))
        fig.add_trace(go.Scatter(x=iterations, y=mean_scores, mode='lines', name=f"{experiment_type}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{experiment_type}", showlegend=True, hovertemplate=f'{experiment_type}<br>Iteration: %{{x}}<br>Score: %{{y:.3f}}<br>Runs: {all_data[5][experiment_type]}<extra></extra>'), row=1, col=1)

# Plot Uniqueness (row 1, col 2)
for idx, experiment_type in enumerate(experiment_types):
    if len(all_data[1][experiment_type]) > 0:
        mean_uniqueness = np.mean(all_data[1][experiment_type], axis=0)
        iterations = list(range(len(mean_uniqueness)))
        fig.add_trace(go.Scatter(x=iterations, y=mean_uniqueness, mode='lines', name=f"{experiment_type}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{experiment_type}", showlegend=False, hovertemplate=f'{experiment_type}<br>Iteration: %{{x}}<br>Uniqueness: %{{y:.3f}}<br>Runs: {all_data[5][experiment_type]}<extra></extra>'), row=1, col=2)

# Plot Num Unique Correct Answers (row 1, col 3)
for idx, experiment_type in enumerate(experiment_types):
    if len(all_data[2][experiment_type]) > 0:
        mean_num_correct = np.mean(all_data[2][experiment_type], axis=0)
        iterations = list(range(len(mean_num_correct)))
        fig.add_trace(go.Scatter(x=iterations, y=mean_num_correct, mode='lines', name=f"{experiment_type}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{experiment_type}", showlegend=False, hovertemplate=f'{experiment_type}<br>Iteration: %{{x}}<br>Count: %{{y:.1f}}<br>Runs: {all_data[5][experiment_type]}<extra></extra>'), row=1, col=3)

# Plot Correct Logprobs (row 2, col 1)
for idx, experiment_type in enumerate(experiment_types):
    if len(all_data[3][experiment_type]) > 0:
        mean_correct_logprobs = np.mean(all_data[3][experiment_type], axis=0)
        iterations = list(range(len(mean_correct_logprobs)))
        fig.add_trace(go.Scatter(x=iterations, y=mean_correct_logprobs, mode='lines', name=f"{experiment_type}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{experiment_type}", showlegend=False, hovertemplate=f'{experiment_type}<br>Iteration: %{{x}}<br>Logprob: %{{y:.3f}}<br>Runs: {all_data[5][experiment_type]}<extra></extra>'), row=2, col=1)

# Plot Incorrect Logprobs (row 2, col 2)
for idx, experiment_type in enumerate(experiment_types):
    if len(all_data[4][experiment_type]) > 0:
        mean_incorrect_logprobs = np.mean(all_data[4][experiment_type], axis=0)
        iterations = list(range(len(mean_incorrect_logprobs)))
        fig.add_trace(go.Scatter(x=iterations, y=mean_incorrect_logprobs, mode='lines', name=f"{experiment_type}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{experiment_type}", showlegend=False, hovertemplate=f'{experiment_type}<br>Iteration: %{{x}}<br>Logprob: %{{y:.3f}}<br>Runs: {all_data[5][experiment_type]}<extra></extra>'), row=2, col=2)

# Update axes labels
for col in range(1, 4):
    fig.update_xaxes(title_text="Iteration", row=1, col=col)
    fig.update_xaxes(title_text="Iteration", row=2, col=col)

fig.update_yaxes(title_text="Mean Eval Score", range=[0, 1], row=1, col=1)
fig.update_yaxes(title_text="Uniqueness", range=[0, 1], row=1, col=2)
fig.update_yaxes(title_text="Count", row=1, col=3)
fig.update_yaxes(title_text="Log Prob", row=2, col=1)
fig.update_yaxes(title_text="Log Prob", row=2, col=2)

# Update layout
fig.update_layout(height=700, hovermode='closest', legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02))

st.plotly_chart(fig, use_container_width=True)

# Display run counts
st.subheader("Run Counts")
for experiment_type in experiment_types:
    count = all_data[5][experiment_type]
    st.markdown(f"- {experiment_type}: **{count}** runs")

# Summary statistics
st.subheader("Summary Statistics")
filtered_results = [res for res in exp_results if (res["task_id"] == selected_task_id or selected_task_id == "all")]

# Apply run filter
filtered_results = [res for res in filtered_results if matches_filter(res, run_filter)]

total_runs = len(filtered_results)
success_runs = len([res for res in filtered_results if res["is_success"]])
success_rate = (success_runs / total_runs * 100) if total_runs > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Runs", total_runs)
col2.metric("Successful Runs", success_runs)
col3.metric("Success Rate", f"{success_rate:.1f}%")

