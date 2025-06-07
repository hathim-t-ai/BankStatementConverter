import os
import io
import csv
import json
import uuid
import tempfile
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
import pandas as pd
import threading
import time

from bankstatementconverter.advanced_converter import AdvancedBankStatementConverter

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store processing jobs in memory (use Redis/database for production)
processing_jobs = {}

def cleanup_old_jobs():
  """Clean up jobs older than 2 hours"""
  cutoff = datetime.now() - timedelta(hours=2)
  jobs_to_remove = []
  
  for job_id, job in processing_jobs.items():
    if job.get('created_at', datetime.now()) < cutoff:
      # Clean up files
      for file_path in job.get('files', []):
        try:
          if os.path.exists(file_path):
            os.remove(file_path)
        except:
          pass
      
      # Clean up output file
      output_file = job.get('output_file')
      if output_file and os.path.exists(output_file):
        try:
          os.remove(output_file)
        except:
          pass
      
      jobs_to_remove.append(job_id)
  
  for job_id in jobs_to_remove:
    processing_jobs.pop(job_id, None)

@app.route('/')
def index():
  # Clean up old jobs on page load
  cleanup_old_jobs()
  return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
  try:
    # Validate request
    if 'files' not in request.files:
      return jsonify({'success': False, 'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files')
    headers = request.form.get('headers', '').strip()
    
    if not headers:
      return jsonify({'success': False, 'error': 'Please specify column headers'}), 400
    
    if not files or all(f.filename == '' for f in files):
      return jsonify({'success': False, 'error': 'No files selected'}), 400
    
    # Validate PDF files
    valid_files = []
    for file in files:
      if file and file.filename.lower().endswith('.pdf'):
        valid_files.append(file)
    
    if not valid_files:
      return jsonify({'success': False, 'error': 'Please upload valid PDF files'}), 400
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded files
    uploaded_files = []
    for file in valid_files:
      filename = secure_filename(file.filename)
      file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
      file.save(file_path)
      uploaded_files.append({
        'path': file_path,
        'original_name': file.filename,
        'size': os.path.getsize(file_path)
      })
    
    # Parse headers
    header_list = [h.strip() for h in headers.split(',') if h.strip()]
    
    # Initialize processing job
    processing_jobs[job_id] = {
      'id': job_id,
      'status': 'processing',
      'files': uploaded_files,
      'headers': header_list,
      'created_at': datetime.now(),
      'progress': 0,
      'message': 'Starting processing...',
      'total_files': len(uploaded_files),
      'processed_files': 0,
      'extracted_records': 0
    }
    
    # Store job ID in session for recovery
    session['current_job_id'] = job_id
    
    # Start processing in background thread
    thread = threading.Thread(target=process_bank_statements, args=(job_id,))
    thread.daemon = True
    thread.start()
    
    return jsonify({
      'success': True,
      'job_id': job_id,
      'message': 'Upload successful, processing started',
      'total_files': len(uploaded_files)
    })
    
  except Exception as e:
    return jsonify({'success': False, 'error': f'Upload failed: {str(e)}'}), 500

def process_bank_statements(job_id):
  """Process bank statements with proper progress tracking"""
  job = processing_jobs.get(job_id)
  if not job:
    return
  
  try:
    # Update status
    job['status'] = 'processing'
    job['message'] = 'Initializing converter...'
    job['progress'] = 5
    
    # Initialize converter
    converter = AdvancedBankStatementConverter(job['headers'])
    
    # Extract file paths
    file_paths = [f['path'] for f in job['files']]
    
    # Update progress
    job['message'] = 'Analyzing PDF structure...'
    job['progress'] = 15
    
    def progress_callback(progress, message):
      """Update job progress during processing"""
      if job_id in processing_jobs:
        processing_jobs[job_id]['progress'] = min(15 + int(progress * 0.7), 85)
        processing_jobs[job_id]['message'] = message
    
    # Process files
    results = converter.convert_multiple(file_paths, progress_callback=progress_callback)
    
    # Update progress
    job['progress'] = 90
    job['message'] = 'Finalizing results...'
    
    if results.empty:
      job['status'] = 'completed'
      job['progress'] = 100
      job['message'] = 'No transaction data found in the uploaded files'
      job['extracted_records'] = 0
      job['results'] = None
    else:
      # Save results to CSV
      output_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_output.csv")
      results.to_csv(output_file, index=False)
      
      # Update job with results
      job['status'] = 'completed'
      job['progress'] = 100
      job['extracted_records'] = len(results)
      job['message'] = f'Successfully extracted {len(results)} transaction records'
      job['output_file'] = output_file
      job['results'] = {
        'total_records': len(results),
        'columns': list(results.columns),
        'preview': results.head(5).to_dict('records'),
        'file_size': os.path.getsize(output_file)
      }
    
    job['completed_at'] = datetime.now()
    
  except Exception as e:
    if job_id in processing_jobs:
      processing_jobs[job_id]['status'] = 'error'
      processing_jobs[job_id]['progress'] = 0
      processing_jobs[job_id]['message'] = f'Processing failed: {str(e)}'
      processing_jobs[job_id]['error'] = str(e)

@app.route('/status/<job_id>')
def get_status(job_id):
  """Get processing status for a job"""
  job = processing_jobs.get(job_id)
  if not job:
    return jsonify({'success': False, 'error': 'Job not found'}), 404
  
  response_data = {
    'success': True,
    'job_id': job_id,
    'status': job['status'],
    'progress': job['progress'],
    'message': job['message'],
    'total_files': job['total_files'],
    'processed_files': job.get('processed_files', 0),
    'extracted_records': job.get('extracted_records', 0)
  }
  
  if job['status'] == 'completed' and job.get('results'):
    response_data['results'] = job['results']
  elif job['status'] == 'error':
    response_data['error'] = job.get('error', 'Unknown error')
  
  return jsonify(response_data)

@app.route('/download/<job_id>')
def download_results(job_id):
  """Download CSV results for a completed job"""
  job = processing_jobs.get(job_id)
  if not job:
    return jsonify({'error': 'Job not found'}), 404
  
  if job['status'] != 'completed':
    return jsonify({'error': 'Job not completed yet'}), 400
  
  if not job.get('output_file') or not os.path.exists(job['output_file']):
    return jsonify({'error': 'Results file not found'}), 404
  
  # Generate a meaningful filename
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  download_name = f'bank_statements_{timestamp}_{job["extracted_records"]}records.csv'
  
  return send_file(
    job['output_file'],
    as_attachment=True,
    download_name=download_name,
    mimetype='text/csv'
  )

@app.route('/preview/<job_id>')
def preview_results(job_id):
  """Preview results for a completed job"""
  job = processing_jobs.get(job_id)
  if not job:
    flash('Job not found', 'error')
    return redirect(url_for('index'))
  
  if job['status'] != 'completed':
    flash('Job not completed yet', 'warning')
    return redirect(url_for('index'))
  
  return render_template('preview.html', job=job)

@app.route('/recover')
def recover_session():
  """Recover session data if page was refreshed during processing"""
  job_id = session.get('current_job_id')
  if job_id and job_id in processing_jobs:
    return jsonify({
      'success': True,
      'job_id': job_id,
      'status': processing_jobs[job_id]['status']
    })
  return jsonify({'success': False})

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0', port=8080) 