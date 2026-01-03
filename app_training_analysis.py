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

def load_experiment_args(experiments_folder):
    args_file = os.path.join(experiments_folder, "experiment_args.json")
    if os.path.exists(args_file):
        with open(args_file, "r") as f:
            return json.load(f)
    return {}

def save_experiment_args(experiments_folder, args_dict):
    args_file = os.path.join(experiments_folder, "experiment_args.json")
    with open(args_file, "w") as f:
        json.dump(args_dict, f, indent=2)

def shorten_task_id(task_id):
    if task_id == "all":
        return "all"
    shortened = task_id.replace("sharded-", "").replace("livecodebench", "lcb")
    return shortened

def load_saved_filters():
    if os.path.exists(SAVED_FILTERS_FILE):
        with open(SAVED_FILTERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_filters_to_file(filters_dict):
    with open(SAVED_FILTERS_FILE, "w") as f:
        json.dump(filters_dict, f, indent=2)

def load_experiment_data(experiments_folder, breakdown_key="experiment_type"):
    experiments = sorted([exp for exp in os.listdir(experiments_folder)])
    necessary_files = ["args.json", "logs.jsonl", "unique_answers.jsonl"]
    
    exp_results = []
    for exp in experiments:
        if any(not os.path.exists(f"{experiments_folder}/{exp}/{f}") for f in necessary_files):
            continue
        
        with open(f"{experiments_folder}/{exp}/args.json", "r") as f:
            exp_args = json.load(f)
        
        learning_rate = exp_args["learning_rate"]
        experiment_type = get_experiment_type(exp_args)

        git_version = exp_args.get("git_version", None)

        sample_strategy = exp_args.get("sample_strategy", None)
        tree_degree = exp_args.get("tree_degree", None)
        tree_depth = exp_args.get("tree_depth", None)
        group_size = exp_args.get("group_size", None)
        
        exp_logs = []
        with open(f"{experiments_folder}/{exp}/logs.jsonl", "r") as f:
            for line in f:
                exp_logs.append(json.loads(line))
        
        task_id = exp_args["task_id"]
        
        with open(f"{experiments_folder}/{exp}/unique_answers.jsonl", "r") as f:
            unique_answers = [json.loads(line) for line in f]
        
        already_seen_answers = set()
        num_new_answers = []
        for answers in unique_answers:
            this_answers = set(answers)
            new_answers = this_answers - already_seen_answers
            num_new_answers.append(len(new_answers))
            already_seen_answers |= this_answers
        
        mean_eval_score = [log["mean_eval_score"] for log in exp_logs]
        
        # Handle uniqueness metrics - backward compatible
        # Check if new multi-level uniqueness metrics exist
        if "uniqueness_ast" in exp_logs[0]:
            uniqueness_raw = [log.get("uniqueness_raw", 100.0)/100.0 for log in exp_logs]
            uniqueness_ws_norm = [log.get("uniqueness_ws_norm", 100.0)/100.0 for log in exp_logs]
            uniqueness_no_comment = [log.get("uniqueness_no_comment", 100.0)/100.0 for log in exp_logs]
            uniqueness_formatted = [log.get("uniqueness_formatted", 100.0)/100.0 for log in exp_logs]
            uniqueness_ast = [log.get("uniqueness_ast", 100.0)/100.0 for log in exp_logs]
            has_multi_uniqueness = True
        else:
            # Backward compatible: use old uniqueness field
            uniqueness_ast = [log.get("uniqueness", 100.0)/100.0 for log in exp_logs]
            uniqueness_raw = uniqueness_ast
            uniqueness_ws_norm = uniqueness_ast
            uniqueness_no_comment = uniqueness_ast
            uniqueness_formatted = uniqueness_ast
            has_multi_uniqueness = False
        
        # Handle logprobs - backward compatible (can be list or float)
        correct_logprobs = []
        incorrect_logprobs = []
        for log in exp_logs:
            clp = log.get("correct_logprobs", np.nan)
            correct_logprobs.append(np.mean(clp) if isinstance(clp, list) and len(clp) > 0 else (clp if not isinstance(clp, list) else np.nan))
            ilp = log.get("incorrect_logprobs", np.nan)
            incorrect_logprobs.append(np.mean(ilp) if isinstance(ilp, list) and len(ilp) > 0 else (ilp if not isinstance(ilp, list) else np.nan))
        
        num_unique_correct_answers = [log["num_unique_correct_answers"] for log in exp_logs]
        
        # Handle keys with/without "mean_" prefix for backward compatibility
        mean_correct_resp_length = [log.get("correct_resp_length", log.get("mean_correct_resp_length", np.nan)) for log in exp_logs]
        mean_incorrect_resp_length = [log.get("incorrect_resp_length", log.get("mean_incorrect_resp_length", np.nan)) for log in exp_logs]
        
        # Handle NLL metrics (new)
        correct_token_nll = []
        incorrect_token_nll = []
        for log in exp_logs:
            ctn = log.get("correct_token_entropy", log.get("correct_token_nll", np.nan))
            correct_token_nll.append(np.mean(ctn) if isinstance(ctn, list) and len(ctn) > 0 else (ctn if not isinstance(ctn, list) else np.nan))
            itn = log.get("incorrect_token_entropy", log.get("incorrect_token_nll", np.nan))
            incorrect_token_nll.append(np.mean(itn) if isinstance(itn, list) and len(itn) > 0 else (itn if not isinstance(itn, list) else np.nan))
        
        is_success = mean_eval_score[-1] >= 0.99
        
        # Get breakdown value based on breakdown_key
        if breakdown_key == "experiment_type":
            breakdown_value = experiment_type
        else:
            breakdown_value = exp_args.get(breakdown_key, f"unknown_{breakdown_key}")
            if breakdown_value is None:
                breakdown_value = f"none_{breakdown_key}"
            else:
                breakdown_value = str(breakdown_value)
        
        exp_results.append({
            "task_id": task_id,
            "mean_eval_score": mean_eval_score,
            "uniqueness_raw": uniqueness_raw,
            "uniqueness_ws_norm": uniqueness_ws_norm,
            "uniqueness_no_comment": uniqueness_no_comment,
            "uniqueness_formatted": uniqueness_formatted,
            "uniqueness_ast": uniqueness_ast,
            "has_multi_uniqueness": has_multi_uniqueness,
            "learning_rate": learning_rate,
            "experiment_type": experiment_type,
            "breakdown_value": breakdown_value,
            "correct_logprobs": correct_logprobs,
            "incorrect_logprobs": incorrect_logprobs,
            "num_unique_correct_answers": num_unique_correct_answers,
            "num_new_answers": num_new_answers,
            "is_success": is_success,
            "sample_strategy": sample_strategy,
            "tree_degree": tree_degree,
            "tree_depth": tree_depth,
            "group_size": group_size,
            "mean_correct_resp_length": mean_correct_resp_length,
            "mean_incorrect_resp_length": mean_incorrect_resp_length,
            "correct_token_nll": correct_token_nll,
            "incorrect_token_nll": incorrect_token_nll
        })
    
    return exp_results

# Find all experiment folders
available_folders = sorted([f for f in os.listdir(".") if f.startswith("experiments") and os.path.isdir(f)])
if not available_folders:
    st.error("No experiment folders found")
    st.stop()

# Sidebar: Experiments folder selector
default_index = available_folders.index("experiments") if "experiments" in available_folders else 0
selected_folder = st.sidebar.selectbox("Experiments Folder", available_folders, index=default_index)

# Load experiment args for this folder
experiment_args = load_experiment_args(selected_folder)
current_breakdown_key = experiment_args.get("breakdown_key", "experiment_type")

# Breakdown key selector
st.sidebar.subheader("Breakdown Settings")
new_breakdown_key = st.sidebar.text_input("Breakdown Key", value=current_breakdown_key, help="The key to group experiments by (default: experiment_type). Examples: learning_rate, tree_degree, sample_strategy")

# Save breakdown key if changed
if new_breakdown_key != current_breakdown_key:
    experiment_args["breakdown_key"] = new_breakdown_key
    save_experiment_args(selected_folder, experiment_args)
    st.sidebar.success(f"Saved breakdown key: {new_breakdown_key}")
    current_breakdown_key = new_breakdown_key

# Load data
try:
    exp_results = load_experiment_data(selected_folder, current_breakdown_key)
    st.sidebar.success(f"Loaded {len(exp_results)} experiments from '{selected_folder}'")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Get unique task_ids and breakdown values
task_ids = sorted(list(set([res["task_id"] for res in exp_results])))
breakdown_values = sorted(list(set([res["breakdown_value"] for res in exp_results])))

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
st.sidebar.divider()
st.sidebar.header("Filters")
task_id_display = ["all"] + [shorten_task_id(tid) for tid in task_ids]
task_id_mapping = {shorten_task_id(tid): tid for tid in task_ids}
task_id_mapping["all"] = "all"
selected_display = st.sidebar.selectbox("Task ID", task_id_display, index=0)
selected_task_id = task_id_mapping[selected_display]
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
filter_info = f"**Task:** {shorten_task_id(selected_task_id)}"
if run_filter:
    filter_info += f" | **Run Filter:** {json.dumps(run_filter)}"
st.markdown(filter_info)

# Process data for plots
def process_data_for_plots(task_id, success_only, max_iterations, run_filter_dict):
    # Define metrics configuration
    metric_keys = ["mean_eval_score", "uniqueness_raw", "uniqueness_ws_norm", "uniqueness_no_comment", "uniqueness_formatted", "uniqueness_ast", "num_unique_correct_answers", "correct_logprobs", "incorrect_logprobs", "mean_correct_resp_length", "mean_incorrect_resp_length", "correct_token_nll", "incorrect_token_nll"]
    
    all_metrics = {key: {} for key in metric_keys}
    run_counts = {}
    
    for breakdown_value in breakdown_values:
        this_results = [res for res in exp_results if (res["task_id"] == task_id or task_id == "all") and res["breakdown_value"] == breakdown_value]
        
        # Apply run filter
        this_results = [res for res in this_results if matches_filter(res, run_filter_dict)]
        
        if success_only:
            this_results = [res for res in this_results if res["is_success"]]
        
        run_counts[breakdown_value] = len(this_results)
        
        # Group by task_id for equal weighting
        task_groups = defaultdict(list)
        for res in this_results:
            task_groups[res["task_id"]].append(res)
        
        # Initialize task-averaged lists for each metric
        task_averaged_metrics = {key: [] for key in metric_keys}
        
        for task_id_key, task_results in task_groups.items():
            # Collect all runs for this task_id
            task_metrics = {key: [] for key in metric_keys}
            
            for res in task_results:
                for key in metric_keys:
                    metric_values = res[key][:max_iterations]
                    if len(metric_values) < max_iterations:
                        metric_values += [metric_values[-1]] * (max_iterations - len(metric_values))
                    task_metrics[key].append(metric_values)
            
            # Average across runs for this task_id
            for key in metric_keys:
                task_averaged_metrics[key].append(np.nanmean(task_metrics[key], axis=0))
        
        # Store the task-averaged data (will be averaged across task_ids in plotting)
        for key in metric_keys:
            all_metrics[key][breakdown_value] = task_averaged_metrics[key]
    
    return all_metrics["mean_eval_score"], all_metrics["uniqueness_raw"], all_metrics["uniqueness_ws_norm"], all_metrics["uniqueness_no_comment"], all_metrics["uniqueness_formatted"], all_metrics["uniqueness_ast"], all_metrics["num_unique_correct_answers"], all_metrics["correct_logprobs"], all_metrics["incorrect_logprobs"], all_metrics["mean_correct_resp_length"], all_metrics["mean_incorrect_resp_length"], all_metrics["correct_token_nll"], all_metrics["incorrect_token_nll"], run_counts

# Process data for all runs
all_data = process_data_for_plots(selected_task_id, False, max_iter, run_filter)

# Check if we have multi-level uniqueness data
has_multi_uniqueness = any(res.get("has_multi_uniqueness", False) for res in exp_results)

# Create subplot layout based on whether we have multi-level uniqueness
if has_multi_uniqueness:
    # 5x3 layout for all metrics including 5 uniqueness measures
    fig = make_subplots(rows=5, cols=3, subplot_titles=("Mean Eval Scores", "Uniqueness Raw", "Uniqueness WS-Norm", "Uniqueness No-Comment", "Uniqueness Formatted", "Uniqueness AST", "Num Unique Correct", "Correct Logprobs", "Incorrect Logprobs", "Correct Resp Length", "Incorrect Resp Length", "Correct Token NLL", "Incorrect Token NLL"), vertical_spacing=0.08, horizontal_spacing=0.08)
else:
    # 3x3 layout for backward compatibility
    fig = make_subplots(rows=3, cols=3, subplot_titles=("Mean Eval Scores", "Uniqueness", "Num Unique Correct Answers", "Correct Logprobs", "Incorrect Logprobs", "Correct Resp Length", "Incorrect Resp Length", "Correct Token NLL", "Incorrect Token NLL"), vertical_spacing=0.12, horizontal_spacing=0.08)

# Colors for experiment types
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

if has_multi_uniqueness:
    # NEW LAYOUT: 5x3 with separate uniqueness plots
    # Row 1: Mean Eval Scores, Uniqueness Raw, Uniqueness WS-Norm
    for idx, breakdown_value in enumerate(breakdown_values):
        if len(all_data[0][breakdown_value]) > 0:
            mean_scores = np.nanmean(all_data[0][breakdown_value], axis=0)
            iterations = list(range(len(mean_scores)))
            fig.add_trace(go.Scatter(x=iterations, y=mean_scores, mode='lines', name=f"{breakdown_value}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{breakdown_value}", showlegend=True, hovertemplate=f'{breakdown_value}<br>Iteration: %{{x}}<br>Score: %{{y:.3f}}<br>Runs: {all_data[13][breakdown_value]}<extra></extra>'), row=1, col=1)
    
    for idx, breakdown_value in enumerate(breakdown_values):
        if len(all_data[1][breakdown_value]) > 0:
            mean_uniqueness = np.nanmean(all_data[1][breakdown_value], axis=0)
            iterations = list(range(len(mean_uniqueness)))
            fig.add_trace(go.Scatter(x=iterations, y=mean_uniqueness, mode='lines', name=f"{breakdown_value}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{breakdown_value}", showlegend=False, hovertemplate=f'{breakdown_value}<br>Iteration: %{{x}}<br>Uniqueness: %{{y:.3f}}<br>Runs: {all_data[13][breakdown_value]}<extra></extra>'), row=1, col=2)
    
    for idx, breakdown_value in enumerate(breakdown_values):
        if len(all_data[2][breakdown_value]) > 0:
            mean_uniqueness = np.nanmean(all_data[2][breakdown_value], axis=0)
            iterations = list(range(len(mean_uniqueness)))
            fig.add_trace(go.Scatter(x=iterations, y=mean_uniqueness, mode='lines', name=f"{breakdown_value}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{breakdown_value}", showlegend=False, hovertemplate=f'{breakdown_value}<br>Iteration: %{{x}}<br>Uniqueness: %{{y:.3f}}<br>Runs: {all_data[13][breakdown_value]}<extra></extra>'), row=1, col=3)
    
    # Row 2: Uniqueness No-Comment, Uniqueness Formatted, Uniqueness AST
    for idx, breakdown_value in enumerate(breakdown_values):
        if len(all_data[3][breakdown_value]) > 0:
            mean_uniqueness = np.nanmean(all_data[3][breakdown_value], axis=0)
            iterations = list(range(len(mean_uniqueness)))
            fig.add_trace(go.Scatter(x=iterations, y=mean_uniqueness, mode='lines', name=f"{breakdown_value}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{breakdown_value}", showlegend=False, hovertemplate=f'{breakdown_value}<br>Iteration: %{{x}}<br>Uniqueness: %{{y:.3f}}<br>Runs: {all_data[13][breakdown_value]}<extra></extra>'), row=2, col=1)
    
    for idx, breakdown_value in enumerate(breakdown_values):
        if len(all_data[4][breakdown_value]) > 0:
            mean_uniqueness = np.nanmean(all_data[4][breakdown_value], axis=0)
            iterations = list(range(len(mean_uniqueness)))
            fig.add_trace(go.Scatter(x=iterations, y=mean_uniqueness, mode='lines', name=f"{breakdown_value}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{breakdown_value}", showlegend=False, hovertemplate=f'{breakdown_value}<br>Iteration: %{{x}}<br>Uniqueness: %{{y:.3f}}<br>Runs: {all_data[13][breakdown_value]}<extra></extra>'), row=2, col=2)
    
    for idx, breakdown_value in enumerate(breakdown_values):
        if len(all_data[5][breakdown_value]) > 0:
            mean_uniqueness = np.nanmean(all_data[5][breakdown_value], axis=0)
            iterations = list(range(len(mean_uniqueness)))
            fig.add_trace(go.Scatter(x=iterations, y=mean_uniqueness, mode='lines', name=f"{breakdown_value}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{breakdown_value}", showlegend=False, hovertemplate=f'{breakdown_value}<br>Iteration: %{{x}}<br>Uniqueness: %{{y:.3f}}<br>Runs: {all_data[13][breakdown_value]}<extra></extra>'), row=2, col=3)
    
    # Row 3: Num Unique Correct, Correct Logprobs, Incorrect Logprobs
    for idx, breakdown_value in enumerate(breakdown_values):
        if len(all_data[6][breakdown_value]) > 0:
            mean_num_correct = np.nanmean(all_data[6][breakdown_value], axis=0)
            iterations = list(range(len(mean_num_correct)))
            fig.add_trace(go.Scatter(x=iterations, y=mean_num_correct, mode='lines', name=f"{breakdown_value}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{breakdown_value}", showlegend=False, hovertemplate=f'{breakdown_value}<br>Iteration: %{{x}}<br>Count: %{{y:.1f}}<br>Runs: {all_data[13][breakdown_value]}<extra></extra>'), row=3, col=1)
    
    for idx, breakdown_value in enumerate(breakdown_values):
        if len(all_data[7][breakdown_value]) > 0:
            mean_correct_logprobs = np.nanmean(all_data[7][breakdown_value], axis=0)
            iterations = list(range(len(mean_correct_logprobs)))
            fig.add_trace(go.Scatter(x=iterations, y=mean_correct_logprobs, mode='lines', name=f"{breakdown_value}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{breakdown_value}", showlegend=False, hovertemplate=f'{breakdown_value}<br>Iteration: %{{x}}<br>Logprob: %{{y:.3f}}<br>Runs: {all_data[13][breakdown_value]}<extra></extra>'), row=3, col=2)
    
    for idx, breakdown_value in enumerate(breakdown_values):
        if len(all_data[8][breakdown_value]) > 0:
            mean_incorrect_logprobs = np.nanmean(all_data[8][breakdown_value], axis=0)
            iterations = list(range(len(mean_incorrect_logprobs)))
            fig.add_trace(go.Scatter(x=iterations, y=mean_incorrect_logprobs, mode='lines', name=f"{breakdown_value}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{breakdown_value}", showlegend=False, hovertemplate=f'{breakdown_value}<br>Iteration: %{{x}}<br>Logprob: %{{y:.3f}}<br>Runs: {all_data[13][breakdown_value]}<extra></extra>'), row=3, col=3)
    
    # Row 4: Correct Resp Length, Incorrect Resp Length, Correct Token NLL
    for idx, breakdown_value in enumerate(breakdown_values):
        if len(all_data[9][breakdown_value]) > 0:
            mean_correct_resp_length = np.nanmean(all_data[9][breakdown_value], axis=0)
            iterations = list(range(len(mean_correct_resp_length)))
            fig.add_trace(go.Scatter(x=iterations, y=mean_correct_resp_length, mode='lines', name=f"{breakdown_value}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{breakdown_value}", showlegend=False, hovertemplate=f'{breakdown_value}<br>Iteration: %{{x}}<br>Length: %{{y:.1f}}<br>Runs: {all_data[13][breakdown_value]}<extra></extra>'), row=4, col=1)
    
    for idx, breakdown_value in enumerate(breakdown_values):
        if len(all_data[10][breakdown_value]) > 0:
            mean_incorrect_resp_length = np.nanmean(all_data[10][breakdown_value], axis=0)
            iterations = list(range(len(mean_incorrect_resp_length)))
            fig.add_trace(go.Scatter(x=iterations, y=mean_incorrect_resp_length, mode='lines', name=f"{breakdown_value}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{breakdown_value}", showlegend=False, hovertemplate=f'{breakdown_value}<br>Iteration: %{{x}}<br>Length: %{{y:.1f}}<br>Runs: {all_data[13][breakdown_value]}<extra></extra>'), row=4, col=2)
    
    for idx, breakdown_value in enumerate(breakdown_values):
        if len(all_data[11][breakdown_value]) > 0:
            mean_correct_token_nll = np.nanmean(all_data[11][breakdown_value], axis=0)
            iterations = list(range(len(mean_correct_token_nll)))
            fig.add_trace(go.Scatter(x=iterations, y=mean_correct_token_nll, mode='lines', name=f"{breakdown_value}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{breakdown_value}", showlegend=False, hovertemplate=f'{breakdown_value}<br>Iteration: %{{x}}<br>NLL: %{{y:.3f}}<br>Runs: {all_data[13][breakdown_value]}<extra></extra>'), row=4, col=3)
    
    # Row 5: Incorrect Token NLL
    for idx, breakdown_value in enumerate(breakdown_values):
        if len(all_data[12][breakdown_value]) > 0:
            mean_incorrect_token_nll = np.nanmean(all_data[12][breakdown_value], axis=0)
            iterations = list(range(len(mean_incorrect_token_nll)))
            fig.add_trace(go.Scatter(x=iterations, y=mean_incorrect_token_nll, mode='lines', name=f"{breakdown_value}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{breakdown_value}", showlegend=False, hovertemplate=f'{breakdown_value}<br>Iteration: %{{x}}<br>NLL: %{{y:.3f}}<br>Runs: {all_data[13][breakdown_value]}<extra></extra>'), row=5, col=1)
    
else:
    # BACKWARD COMPATIBLE LAYOUT: 3x3
    # Plot Mean Eval Scores (row 1, col 1)
    for idx, breakdown_value in enumerate(breakdown_values):
        if len(all_data[0][breakdown_value]) > 0:
            mean_scores = np.nanmean(all_data[0][breakdown_value], axis=0)
            iterations = list(range(len(mean_scores)))
            fig.add_trace(go.Scatter(x=iterations, y=mean_scores, mode='lines', name=f"{breakdown_value}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{breakdown_value}", showlegend=True, hovertemplate=f'{breakdown_value}<br>Iteration: %{{x}}<br>Score: %{{y:.3f}}<br>Runs: {all_data[13][breakdown_value]}<extra></extra>'), row=1, col=1)

    # Plot Uniqueness (row 1, col 2) - single uniqueness (AST level)
    for idx, breakdown_value in enumerate(breakdown_values):
        if len(all_data[5][breakdown_value]) > 0:
            mean_uniqueness = np.nanmean(all_data[5][breakdown_value], axis=0)
            iterations = list(range(len(mean_uniqueness)))
            fig.add_trace(go.Scatter(x=iterations, y=mean_uniqueness, mode='lines', name=f"{breakdown_value}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{breakdown_value}", showlegend=False, hovertemplate=f'{breakdown_value}<br>Iteration: %{{x}}<br>Uniqueness: %{{y:.3f}}<br>Runs: {all_data[13][breakdown_value]}<extra></extra>'), row=1, col=2)

    # Plot Num Unique Correct Answers (row 1, col 3)
    for idx, breakdown_value in enumerate(breakdown_values):
        if len(all_data[6][breakdown_value]) > 0:
            mean_num_correct = np.nanmean(all_data[6][breakdown_value], axis=0)
            iterations = list(range(len(mean_num_correct)))
            fig.add_trace(go.Scatter(x=iterations, y=mean_num_correct, mode='lines', name=f"{breakdown_value}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{breakdown_value}", showlegend=False, hovertemplate=f'{breakdown_value}<br>Iteration: %{{x}}<br>Count: %{{y:.1f}}<br>Runs: {all_data[13][breakdown_value]}<extra></extra>'), row=1, col=3)

    # Plot Correct Logprobs (row 2, col 1)
    for idx, breakdown_value in enumerate(breakdown_values):
        if len(all_data[7][breakdown_value]) > 0:
            mean_correct_logprobs = np.nanmean(all_data[7][breakdown_value], axis=0)
            iterations = list(range(len(mean_correct_logprobs)))
            fig.add_trace(go.Scatter(x=iterations, y=mean_correct_logprobs, mode='lines', name=f"{breakdown_value}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{breakdown_value}", showlegend=False, hovertemplate=f'{breakdown_value}<br>Iteration: %{{x}}<br>Logprob: %{{y:.3f}}<br>Runs: {all_data[13][breakdown_value]}<extra></extra>'), row=2, col=1)

    # Plot Incorrect Logprobs (row 2, col 2)
    for idx, breakdown_value in enumerate(breakdown_values):
        if len(all_data[8][breakdown_value]) > 0:
            mean_incorrect_logprobs = np.nanmean(all_data[8][breakdown_value], axis=0)
            iterations = list(range(len(mean_incorrect_logprobs)))
            fig.add_trace(go.Scatter(x=iterations, y=mean_incorrect_logprobs, mode='lines', name=f"{breakdown_value}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{breakdown_value}", showlegend=False, hovertemplate=f'{breakdown_value}<br>Iteration: %{{x}}<br>Logprob: %{{y:.3f}}<br>Runs: {all_data[13][breakdown_value]}<extra></extra>'), row=2, col=2)

    # Plot Correct Resp Length (row 2, col 3)
    for idx, breakdown_value in enumerate(breakdown_values):
        if len(all_data[9][breakdown_value]) > 0:
            mean_correct_resp_length = np.nanmean(all_data[9][breakdown_value], axis=0)
            iterations = list(range(len(mean_correct_resp_length)))
            fig.add_trace(go.Scatter(x=iterations, y=mean_correct_resp_length, mode='lines', name=f"{breakdown_value}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{breakdown_value}", showlegend=False, hovertemplate=f'{breakdown_value}<br>Iteration: %{{x}}<br>Length: %{{y:.1f}}<br>Runs: {all_data[13][breakdown_value]}<extra></extra>'), row=2, col=3)

    # Plot Incorrect Resp Length (row 3, col 1)
    for idx, breakdown_value in enumerate(breakdown_values):
        if len(all_data[10][breakdown_value]) > 0:
            mean_incorrect_resp_length = np.nanmean(all_data[10][breakdown_value], axis=0)
            iterations = list(range(len(mean_incorrect_resp_length)))
            fig.add_trace(go.Scatter(x=iterations, y=mean_incorrect_resp_length, mode='lines', name=f"{breakdown_value}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{breakdown_value}", showlegend=False, hovertemplate=f'{breakdown_value}<br>Iteration: %{{x}}<br>Length: %{{y:.1f}}<br>Runs: {all_data[13][breakdown_value]}<extra></extra>'), row=3, col=1)

    # Plot Correct Token NLL (row 3, col 2)
    for idx, breakdown_value in enumerate(breakdown_values):
        if len(all_data[11][breakdown_value]) > 0:
            mean_correct_token_nll = np.nanmean(all_data[11][breakdown_value], axis=0)
            iterations = list(range(len(mean_correct_token_nll)))
            fig.add_trace(go.Scatter(x=iterations, y=mean_correct_token_nll, mode='lines', name=f"{breakdown_value}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{breakdown_value}", showlegend=False, hovertemplate=f'{breakdown_value}<br>Iteration: %{{x}}<br>NLL: %{{y:.3f}}<br>Runs: {all_data[13][breakdown_value]}<extra></extra>'), row=3, col=2)

    # Plot Incorrect Token NLL (row 3, col 3)
    for idx, breakdown_value in enumerate(breakdown_values):
        if len(all_data[12][breakdown_value]) > 0:
            mean_incorrect_token_nll = np.nanmean(all_data[12][breakdown_value], axis=0)
            iterations = list(range(len(mean_incorrect_token_nll)))
            fig.add_trace(go.Scatter(x=iterations, y=mean_incorrect_token_nll, mode='lines', name=f"{breakdown_value}", line=dict(color=colors[idx % len(colors)]), legendgroup=f"exp{breakdown_value}", showlegend=False, hovertemplate=f'{breakdown_value}<br>Iteration: %{{x}}<br>NLL: %{{y:.3f}}<br>Runs: {all_data[13][breakdown_value]}<extra></extra>'), row=3, col=3)

# Update axes labels and layout based on whether we have multi-level uniqueness
if has_multi_uniqueness:
    # 5x3 layout
    for col in range(1, 4):
        for row in range(1, 6):
            fig.update_xaxes(title_text="Iteration", row=row, col=col)
    
    fig.update_yaxes(title_text="Score", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Uniqueness", range=[0, 1], row=1, col=2)
    fig.update_yaxes(title_text="Uniqueness", range=[0, 1], row=1, col=3)
    fig.update_yaxes(title_text="Uniqueness", range=[0, 1], row=2, col=1)
    fig.update_yaxes(title_text="Uniqueness", range=[0, 1], row=2, col=2)
    fig.update_yaxes(title_text="Uniqueness", range=[0, 1], row=2, col=3)
    fig.update_yaxes(title_text="Count", row=3, col=1)
    fig.update_yaxes(title_text="Log Prob", row=3, col=2)
    fig.update_yaxes(title_text="Log Prob", row=3, col=3)
    fig.update_yaxes(title_text="Length", row=4, col=1)
    fig.update_yaxes(title_text="Length", row=4, col=2)
    fig.update_yaxes(title_text="NLL", row=4, col=3)
    fig.update_yaxes(title_text="NLL", row=5, col=1)
    
    fig.update_layout(height=1400, hovermode='closest', legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02))
else:
    # 3x3 layout
    for col in range(1, 4):
        fig.update_xaxes(title_text="Iteration", row=1, col=col)
        fig.update_xaxes(title_text="Iteration", row=2, col=col)
        fig.update_xaxes(title_text="Iteration", row=3, col=col)

    fig.update_yaxes(title_text="Mean Eval Score", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Uniqueness", range=[0, 1], row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=3)
    fig.update_yaxes(title_text="Log Prob", row=2, col=1)
    fig.update_yaxes(title_text="Log Prob", row=2, col=2)
    fig.update_yaxes(title_text="Length", row=2, col=3)
    fig.update_yaxes(title_text="Length", row=3, col=1)
    fig.update_yaxes(title_text="NLL", row=3, col=2)
    fig.update_yaxes(title_text="NLL", row=3, col=3)

    fig.update_layout(height=900, hovermode='closest', legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02))

st.plotly_chart(fig, use_container_width=True)

# Create scatter plot: Uniqueness vs Mean Eval Score (all iterations, averaged by task_id)
st.subheader("Uniqueness vs Mean Eval Score (All Iterations)")
scatter_fig = go.Figure()

# Helper function to convert hex color to rgba
def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'

for idx, breakdown_value in enumerate(breakdown_values):
    this_results = [res for res in exp_results if (res["task_id"] == selected_task_id or selected_task_id == "all") and res["breakdown_value"] == breakdown_value]
    this_results = [res for res in this_results if matches_filter(res, run_filter)]
    
    if len(this_results) > 0:
        # Group by task_id
        task_groups = defaultdict(list)
        for res in this_results:
            task_groups[res["task_id"]].append(res)
        
        base_color = colors[idx % len(colors)]
        line_color = hex_to_rgba(base_color, 0.3)
        
        # Collect all scatter points and compute average trajectory
        all_uniqueness_by_iter = [[] for _ in range(max_iter)]
        all_scores_by_iter = [[] for _ in range(max_iter)]
        
        # For each task_id, average across runs and collect points
        for task_id, task_results in task_groups.items():
            for iter_idx in range(max_iter):
                # Use AST-level uniqueness for scatter plot
                uniqueness_at_iter = [res["uniqueness_ast"][iter_idx] if iter_idx < len(res["uniqueness_ast"]) else res["uniqueness_ast"][-1] for res in task_results]
                score_at_iter = [res["mean_eval_score"][iter_idx] if iter_idx < len(res["mean_eval_score"]) else res["mean_eval_score"][-1] for res in task_results]
                
                avg_uniqueness = np.nanmean(uniqueness_at_iter)
                avg_score = np.nanmean(score_at_iter)
                
                all_uniqueness_by_iter[iter_idx].append(avg_uniqueness)
                all_scores_by_iter[iter_idx].append(avg_score)
        
        # Flatten for scatter plot
        all_scatter_x = [val for iter_vals in all_uniqueness_by_iter for val in iter_vals]
        all_scatter_y = [val for iter_vals in all_scores_by_iter for val in iter_vals]
        
        # Add scatter points for all task_ids
        scatter_fig.add_trace(go.Scatter(x=all_scatter_x, y=all_scatter_y, mode='markers', name=f"{breakdown_value}", marker=dict(color=base_color, size=6, opacity=0.4), legendgroup=f"exp{breakdown_value}", showlegend=True, hovertemplate=f'{breakdown_value}<br>Uniqueness: %{{x:.3f}}<br>Score: %{{y:.3f}}<extra></extra>'))
        
        # Compute average trajectory across all task_ids
        avg_trajectory_x = [np.nanmean(vals) for vals in all_uniqueness_by_iter]
        avg_trajectory_y = [np.nanmean(vals) for vals in all_scores_by_iter]
        
        # Add average trajectory line
        scatter_fig.add_trace(go.Scatter(x=avg_trajectory_x, y=avg_trajectory_y, mode='lines', name=f"{breakdown_value} (avg)", line=dict(color=base_color, width=2.5), legendgroup=f"exp{breakdown_value}", showlegend=False, hovertemplate=f'{breakdown_value} avg<br>Uniqueness: %{{x:.3f}}<br>Score: %{{y:.3f}}<extra></extra>'))
        
        # Add arrow markers at intervals to show direction
        arrow_interval = max(1, len(avg_trajectory_x) // 5)
        for i in range(arrow_interval, len(avg_trajectory_x), arrow_interval):
            if i > 0:
                scatter_fig.add_annotation(x=avg_trajectory_x[i], y=avg_trajectory_y[i], ax=avg_trajectory_x[i-1], ay=avg_trajectory_y[i-1], xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor=base_color)

scatter_fig.update_layout(xaxis_title="Uniqueness", yaxis_title="Mean Eval Score", height=500, hovermode='closest', legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02))
scatter_fig.update_xaxes(range=[0, 1])
scatter_fig.update_yaxes(range=[0, 1])

st.plotly_chart(scatter_fig, use_container_width=True)

# Display run counts
st.subheader("Run Counts")
for breakdown_value in breakdown_values:
    count = all_data[13][breakdown_value]
    st.markdown(f"- {breakdown_value}: **{count}** runs")

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

