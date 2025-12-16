from flask import Flask, request, jsonify
from evaluation_service import EvaluationService
from utils import print_colored
import multiprocessing
from utils import calculate_gpu_concurrency
import logging, os

app = Flask(__name__)

# Configure logging to suppress check_evaluation_job requests
@app.after_request
def log_request_info(response):
    """Custom logging that skips check_evaluation_job requests"""
    if request.endpoint != 'check_evaluation_job':
        app.logger.info('%s %s %s %s %s', request.remote_addr, request.method, request.url, response.status_code, response.content_length or '-')
    return response

# Disable the default Flask request logging since we're handling it ourselves
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Global variable to hold the service instance
evaluation_service = None

@app.route('/shutdown_service', methods=['POST'])
def shutdown_service():
    """Shutdown the evaluation service"""
    global evaluation_service
    
    try:
        if evaluation_service is None:
            return jsonify({
                'status': 'warning',
                'message': 'No evaluation service currently running'
            })
        
        # Shutdown the service
        evaluation_service.shutdown()
        evaluation_service = None
        
        return jsonify({
            'status': 'success',
            'message': 'Evaluation service shut down successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/schedule_evaluation', methods=['POST'])
def schedule_evaluation():
    """Schedule an evaluation job"""
    global evaluation_service
    
    try:
        if evaluation_service is None:
            return jsonify({
                'error': 'Evaluation service not available'
            }), 503
        
        # Get evaluation parameters from request
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'JSON body is required'
            }), 400
        
        required_fields = ['conversation', 'task_name', 'sample']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'{field} parameter is required in JSON body'
                }), 400
        
        conversation = data['conversation']
        task_name = data['task_name']
        sample = data['sample']
        
        # Schedule the evaluation job
        job_id = evaluation_service.schedule_evaluation(
            conversation=conversation,
            task_name=task_name,
            sample=sample
        )
        
        return jsonify({
            'status': 'success',
            'job_id': job_id,
            'message': 'Evaluation job scheduled successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/schedule_evaluation_batch', methods=['POST'])
def schedule_evaluation_batch():
    """Schedule multiple evaluation jobs in batch"""
    global evaluation_service
    
    try:
        if evaluation_service is None:
            return jsonify({
                'error': 'Evaluation service not available'
            }), 503
        
        # Get evaluation parameters from request
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'JSON body is required'
            }), 400
        
        if 'evaluations' not in data:
            return jsonify({
                'error': 'evaluations parameter is required in JSON body'
            }), 400
        
        evaluations = data['evaluations']
        if not isinstance(evaluations, list):
            return jsonify({
                'error': 'evaluations must be a list'
            }), 400
        
        # Validate each evaluation has required fields
        required_fields = ['conversation', 'task_name', 'sample']
        for i, eval_data in enumerate(evaluations):
            for field in required_fields:
                if field not in eval_data:
                    return jsonify({
                        'error': f'Evaluation {i}: {field} parameter is required'
                    }), 400
        
        # Schedule all evaluation jobs
        job_ids = []
        for eval_data in evaluations:
            job_id = evaluation_service.schedule_evaluation(
                conversation=eval_data['conversation'],
                task_name=eval_data['task_name'],
                sample=eval_data['sample']
            )
            job_ids.append(job_id)
        
        return jsonify({
            'status': 'success',
            'job_ids': job_ids,
            'message': f'Successfully scheduled {len(job_ids)} evaluation jobs'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/check_evaluation_job', methods=['GET'])
def check_evaluation_job():
    """Check the status of an evaluation job"""
    global evaluation_service
    
    try:
        if evaluation_service is None:
            return jsonify({
                'error': 'Evaluation service not available'
            }), 503
        
        # Get job_id from query parameters
        job_id = request.args.get('job_id')
        if not job_id:
            return jsonify({
                'error': 'job_id parameter is required'
            }), 400
        
        # Check job status
        result = evaluation_service.check_evaluation_job(job_id)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/check_evaluation_job_batch', methods=['POST'])
def check_evaluation_job_batch():
    """Check the status of multiple evaluation jobs in batch"""
    global evaluation_service
    
    try:
        if evaluation_service is None:
            return jsonify({
                'error': 'Evaluation service not available'
            }), 503
        
        # Get job_ids from request body
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'JSON body is required'
            }), 400
        
        if 'job_ids' not in data:
            return jsonify({
                'error': 'job_ids parameter is required in JSON body'
            }), 400
        
        job_ids = data['job_ids']
        if not isinstance(job_ids, list):
            return jsonify({
                'error': 'job_ids must be a list'
            }), 400
        
        # Check status for all jobs
        results = {}
        for job_id in job_ids:
            results[job_id] = evaluation_service.check_evaluation_job(job_id)
        
        return jsonify({
            'status': 'success',
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get overall service status"""
    global evaluation_service
    
    try:
        if evaluation_service is None:
            return jsonify({
                'service_initialized': False,
                'message': 'Evaluation service not available'
            })
        
        status = evaluation_service.get_status()
        worker_status = evaluation_service.get_worker_status()
        
        return jsonify({
            'service_initialized': True,
            'service_status': status,
            'worker_status': worker_status
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service_initialized': evaluation_service is not None,
        'tasks_loaded': evaluation_service is not None and bool(evaluation_service.tasks_dict) if evaluation_service else False
    })

# Cleanup function to ensure proper shutdown
def cleanup():
    """Clean up resources on shutdown"""
    global evaluation_service
    if evaluation_service is not None:
        evaluation_service.shutdown()
        evaluation_service = None

# Register cleanup function
import atexit
atexit.register(cleanup)

if __name__ == '__main__':
    # Set multiprocessing start method (important for worker processes)
    multiprocessing.set_start_method('spawn', force=True)
    
    print("Starting Flask Evaluation Service...")
    
    # Get number of workers from environment variable or use default

    num_workers = int(os.environ.get('EVAL_WORKERS', 200)) # calculate_gpu_concurrency()["total_concurrency"])
    
    # Initialize evaluation service automatically on startup
    print_colored(f"Initializing evaluation service with {num_workers} workers...", "yellow")
    try:
        evaluation_service = EvaluationService(num_workers=num_workers)
        print_colored("✓ Evaluation service initialized and ready!", "green")
    except Exception as e:
        print_colored(f"✗ Failed to initialize evaluation service: {e}", "red")
        exit(1)
    
    print("Available endpoints:")
    print("  POST /shutdown_service - Shutdown the evaluation service")
    print("  POST /schedule_evaluation - Schedule an evaluation job")
    print("  POST /schedule_evaluation_batch - Schedule multiple evaluation jobs")
    print("  GET /check_evaluation_job?job_id=<id> - Check evaluation job status")
    print("  POST /check_evaluation_job_batch - Check multiple evaluation jobs status")
    print("  GET /status - Get service status")
    print("  GET /health - Health check")
    print(f"\nEvaluation service running with {num_workers} workers (set EVAL_WORKERS env var to change)")
    print("Starting server on http://localhost:5001")
    
    app.run(host='localhost', port=5001, debug=True, use_reloader=False) 