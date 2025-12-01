import streamlit as st
import plotly.graph_objects as go
import os, ujson as json

st.set_page_config(page_title="Training Analysis", layout="wide")

st.title("Training Analysis Dashboard")

# Load experiment data
def load_experiment_data():
    experiments = sorted([exp for exp in os.listdir("../mtco_old/experiments") if exp.startswith("exp")])
    
    exp_results = []
    for exp in experiments:
        if not os.path.exists(f"../mtco_old/experiments/{exp}/args.json") or not os.path.exists(f"../mtco_old/experiments/{exp}/logs.jsonl"):
            continue
        
        with open(f"../mtco_old/experiments/{exp}/args.json", "r") as f:
            exp_args = json.load(f)
        
        learning_rate = exp_args["learning_rate"]
        if "degree" not in exp_args:
            continue
        degree = exp_args["degree"]
        
        exp_logs = []
        with open(f"../mtco_old/experiments/{exp}/logs.jsonl", "r") as f:
            for line in f:
                exp_logs.append(json.loads(line))
        
        task_id = exp_args["task_id"]
        
        mean_eval_score = [log["mean_eval_score"] for log in exp_logs]
        if mean_eval_score[-1] < 0.99:
            continue
        uniqueness = [log["uniqueness"]/100.0 for log in exp_logs]
        
        exp_results.append({"task_id": task_id, "mean_eval_score": mean_eval_score, "uniqueness": uniqueness, "learning_rate": learning_rate, "degree": degree})
    
    return exp_results

# Load data
exp_results = load_experiment_data()

# Get unique values for filters
all_learning_rates = sorted(list(set([result["learning_rate"] for result in exp_results])))
all_degrees = sorted(list(set([result["degree"] for result in exp_results])))

# Sidebar filters
st.sidebar.header("Filters")

selected_learning_rate = st.sidebar.selectbox("Learning Rate", options=["All"] + all_learning_rates, index=0)

selected_degree = st.sidebar.selectbox("Degree", options=["All"] + all_degrees, index=0)

# Filter results
filtered_results = []
for result in exp_results:
    if selected_learning_rate != "All" and result["learning_rate"] != selected_learning_rate:
        continue
    if selected_degree != "All" and result["degree"] != selected_degree:
        continue
    filtered_results.append(result)

# Display statistics
st.sidebar.markdown("---")
st.sidebar.metric("Total Experiments", len(exp_results))
st.sidebar.metric("Filtered Experiments", len(filtered_results))

# Main content
if len(filtered_results) == 0:
    st.warning("No experiments match the selected filters.")
else:
    # Create plots in 3-column grid
    num_cols = 3
    
    for idx, result in enumerate(filtered_results):
        if idx % num_cols == 0:
            cols = st.columns(num_cols)
        
        with cols[idx % num_cols]:
            fig = go.Figure()
            
            iterations = list(range(len(result["mean_eval_score"])))
            
            fig.add_trace(go.Scatter(x=iterations, y=result["mean_eval_score"], mode='lines', name='Mean Eval Score', line=dict(color='blue')))
            
            fig.add_trace(go.Scatter(x=iterations, y=result["uniqueness"], mode='lines', name='Uniqueness', line=dict(color='orange')))
            
            fig.update_layout(title=f"Task {result['task_id']}<br>LR={result['learning_rate']}, Deg={result['degree']}", xaxis_title="Iteration", yaxis_title="Score", yaxis_range=[0, 1], height=400, showlegend=True)
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Show detailed info
    with st.expander("Detailed Results"):
        for result in filtered_results:
            st.write(f"**Task {result['task_id']}**: Learning Rate = {result['learning_rate']}, Degree = {result['degree']}, Final Eval Score = {result['mean_eval_score'][-1]:.4f}, Final Uniqueness = {result['uniqueness'][-1]:.4f}")
