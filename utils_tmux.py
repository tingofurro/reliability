import subprocess, time


def check_session_exists(session_name):
    """Check if a tmux session exists"""
    try:
        result = subprocess.run(['tmux', 'has-session', '-t', session_name], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        print("Error: tmux is not installed or not in PATH")
        return False


def delete_session(session_name):
    """Delete a tmux session if it exists"""
    if not check_session_exists(session_name):
        print(f"Session '{session_name}' does not exist")
        return False
    
    try:
        result = subprocess.run(['tmux', 'kill-session', '-t', session_name],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Session '{session_name}' deleted successfully")
            return True
        else:
            print(f"Error deleting session '{session_name}': {result.stderr}")
            return False
    except FileNotFoundError:
        print("Error: tmux is not installed or not in PATH")
        return False


def create_session(session_name, detached=True, start_directory=None):
    """Create a new tmux session"""
    if check_session_exists(session_name):
        print(f"Session '{session_name}' already exists")
        return False
    
    cmd = ['tmux', 'new-session', '-s', session_name]
    
    if detached:
        cmd.append('-d')
    
    if start_directory:
        cmd.extend(['-c', start_directory])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Session '{session_name}' created successfully")
            return True
        else:
            print(f"Error creating session '{session_name}': {result.stderr}")
            return False
    except FileNotFoundError:
        print("Error: tmux is not installed or not in PATH")
        return False


def launch_command(session_name, command, window_name=None, pane_index=0):
    """Launch a command in a tmux session"""
    if not check_session_exists(session_name):
        print(f"Session '{session_name}' does not exist")
        return False
    
    # Construct the target (session:window.pane)
    target = session_name
    if window_name:
        target += f":{window_name}"
    if pane_index is not None:
        target += f".{pane_index}"
    
    try:
        result = subprocess.run(['tmux', 'send-keys', '-t', target, command, 'Enter'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Command '{command}' sent to session '{session_name}'")
            return True
        else:
            print(f"Error sending command to session '{session_name}': {result.stderr}")
            return False
    except FileNotFoundError:
        print("Error: tmux is not installed or not in PATH")
        return False


def list_sessions():
    """List all existing tmux sessions"""
    try:
        result = subprocess.run(['tmux', 'list-sessions'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip().split('\n') if result.stdout.strip() else []
        else:
            return []
    except FileNotFoundError:
        print("Error: tmux is not installed or not in PATH")
        return []


def attach_session(session_name):
    """Attach to an existing tmux session (interactive)"""
    if not check_session_exists(session_name):
        print(f"Session '{session_name}' does not exist")
        return False
    
    try:
        # Use subprocess.call for interactive attachment
        result = subprocess.call(['tmux', 'attach-session', '-t', session_name])
        return result == 0
    except FileNotFoundError:
        print("Error: tmux is not installed or not in PATH")
        return False

def start_gen_and_eval_sessions():
    # first genserv: if "gen" exists, then kill it first
    if check_session_exists("gen"):
        delete_session("gen")
    create_session("gen")
    # also need to go in ~/mtco_old/
    launch_command("gen", "cd ~/reliability/ && python genserv_app.py")

    # then evalserv: if "eval" exists, then kill it first
    if check_session_exists("eval"):
        delete_session("eval")
    create_session("eval")
    launch_command("eval", "cd ~/reliability/ && python evalserv_app.py")
    time.sleep(20) # 10 seconds is not enough to start genserv, leads to failure

if __name__ == "__main__":
    # Example usage

    # 
    start_gen_and_eval_sessions()
    # session_name = "test_session"
    
    # print("=== Tmux Utils Demo ===")
    
    # # List existing sessions
    # sessions = list_sessions()
    # print(f"Existing sessions: {sessions}")
    
    # # Check if session exists
    # exists = check_session_exists(session_name)
    # print(f"Session '{session_name}' exists: {exists}")
    
    # # Delete session if it exists
    # if exists:
    #     delete_session(session_name)
    
    # # Create new session
    # create_session(session_name)
    
    # # Launch a command
    # launch_command(session_name, "echo 'Hello from tmux!'")
    # launch_command(session_name, "ls -la")
    
    # # List sessions again
    # sessions = list_sessions()
    # print(f"Sessions after creation: {sessions}")
