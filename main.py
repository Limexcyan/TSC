from flask import Flask, request, send_file, render_template, abort, redirect, url_for, send_from_directory, make_response
import os
from io import BytesIO
from jobs import JobsManager
from fileManager import FileManager
from flask_socketio import SocketIO, emit
from datetime import datetime
from datetime import timedelta

app = Flask(__name__)
sio = SocketIO(app)
filemngr = FileManager('D:\\tsclocal-master\\tsclocal\\appdata')
jobs = JobsManager(32,filemngr,app.app_context())

def cold_start_jobs():
    dead_jobs = filemngr.query_processing_manifest() + filemngr.query_jobless_manifest()
    if(len(dead_jobs) < 1):
        return
    print(f"[main] - cold starting {len(dead_jobs)} jobs")
    print(dead_jobs)
    for job in dead_jobs[::-1]:
        token = jobs.allocate_known_job(job["_job"]["job_id"], job["_manifest"])
        jobs.start_job_from_manifest(filemngr,token)
    print("[main] - revived and enqueued dead jobs")

cold_start_jobs()


@sio.on('job_query')
def respond_job_query(data):
    print(f"received query {data}")
    data = data['data']
    response = dict()
    for v in data:
        response[v] = filemngr.open_manifest(v)
    with app.app_context():
        emit('job_query_response', response)

@app.errorhandler(404)
def serve_404(error):
    return render_template("error.html", code404=True)

@app.errorhandler(410)
def serve_410(error):
    return render_template("error.html", code410=True)

@app.errorhandler(500)
def serve_500(error):
    return render_template("error.html", code500=True)

@sio.on("query_job_status")
def serve_query_job_status(data):
    status = jobs.get_job_status(data['data'])
    with app.app_context():
        emit("query_job_status_response", {'token':data['data'],'status':status}, namespace='/')

@app.route("/")
def serve_index():
    return render_template("index.html")

@app.route('/favicon.ico')
def serve_favicon():
    return send_from_directory(os.path.dirname(os.path.realpath(__file__)) + "/resources",'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route("/<mode>")
def serve_mode(mode):
    document = ""
    try:
        document_file = open(os.path.dirname(os.path.realpath(__file__)) + f"/pages/{mode}.html",'r')
        document = document_file.read()
    except Exception:
        abort(500)
    finally:
        document_file.close()
    return document

@app.route("/details/<job_id>")
def serve_details(job_id):
    return render_template("details.html",job_id=job_id,table=filemngr.pretty_print_statuses(job_id))

@app.route("/download/<job_id>/<file_src>/<file_name>")
def serve_download(job_id,file_src,file_name):
    job_id = int(job_id)
    relevant_files = filemngr.open_manifest(job_id)["files"][file_src]
    _file = [_fn["filename"].replace(filemngr.OBFUSCATE_ROOT,filemngr.appdata_path,1) for _fn in relevant_files if file_name in _fn["filename"]]
    return send_file(_file[0],download_name=file_name,as_attachment=True)

@app.route("/analyze/<manifest_token>/<job_id>")
def serve_analysis(manifest_token,job_id):
    job_id = int(job_id)
    manifest = filemngr.open_manifest(manifest_token)
    if(manifest["jobs"][job_id]["mode"] == 'analyze'):
        analysis_infix = [fo["filename"].replace(filemngr.OBFUSCATE_ROOT, filemngr.appdata_path) for fo in manifest["files"]["obj_analyze_graph"] if fo["job_id"] == job_id][0]
        analysis_infix_handle = open(analysis_infix,"r")
        analysis_infix_content = analysis_infix_handle.read()
        analysis_infix_handle.close()
        return render_template("analysis.html",fig_bar_stacked=analysis_infix_content)
    else:
        new_job_id = filemngr.prepare_for_analysis(manifest_token,job_id)
        job_token = jobs.allocate_known_job(new_job_id,manifest_token)
        manifest = filemngr.open_manifest(manifest_token)
        manifest["jobs"].append({'job_id':new_job_id,'status':"allocated",'mode':'analyze','has_analysis':True,'kwargs':{}})
        filemngr.overwrite_manifest(manifest_token,manifest)
        jobs.start_job_from_manifest(filemngr, job_token)
        response = make_response(redirect(url_for('serve_index')))
        cookie_expiration = datetime.now()
        cookie_expiration = cookie_expiration + timedelta(days=3)
        response.set_cookie(f'{manifest_token}',manifest_token,expires=cookie_expiration)
        return response

@app.route("/recluster/<manifest_token>/<job_id>", methods=['GET'])
def serve_recluster(manifest_token,job_id):    
    job_id = int(job_id)
    mfp_radius, mfp_size, cluster_k = filemngr.get_kwargs_for_processing(manifest_token, job_id)
    return render_template("recluster.html",mfp_size=mfp_size, mfp_radius=mfp_radius, cluster_k=cluster_k)

@app.route("/recluster/<manifest_token>/<job_id>", methods=['POST'])
def process_recluster(manifest_token,job_id):
    job_id = int(job_id)
    new_job_id = filemngr.prepare_for_reclustering(manifest_token,job_id)
    job_token = jobs.allocate_known_job(new_job_id,manifest_token)
    files = request.files.to_dict(flat=False)['file']
    file_names = []
    for _file in files:
        filename_save = filemngr.get_file_save_location(manifest_token,_file.filename)
        file_names.append([{"key":"origin","value":_file.filename}])
        filemngr.save_file(manifest_token,new_job_id,'chembl',_file,file_origin=FileManager.FILE_TYPE_WEB_REQUESTS)
        _file.close()
    
    manifest = filemngr.open_manifest(manifest_token)
    job_identity_kwargs = dict()
    job_identity_kwargs['user_append']   = file_names
    job_identity_kwargs['mfp_radius']    = int(request.form['mfp-radius'][:-1])
    job_identity_kwargs['mfp_size']      = int(request.form['mfp-size'][:-1])
    job_identity_kwargs['cluster_count'] = int(request.form['cluster-k'][:-1])
    manifest["jobs"].append({'job_id':new_job_id,'status':"allocated",'mode':'recluster','has_analysis':False,'kwargs':job_identity_kwargs})
    filemngr.overwrite_manifest(manifest_token,manifest)
    request.close()
    jobs.start_job_from_manifest(filemngr,job_token)
    response = make_response(redirect(url_for('serve_index')))
    cookie_expiration = datetime.now()
    cookie_expiration = cookie_expiration + timedelta(days=3)
    response.set_cookie(f'{manifest_token}',manifest_token,expires=cookie_expiration)
    return response

@app.route("/action", methods=['POST'])
def serve_action():
    
    job_identity_current_job_id = 0
    allocate_job_key = jobs.allocate_job(job_identity_current_job_id)
    filemngr.create_directory(allocate_job_key)
    filemngr.create_manifest(allocate_job_key)
    
    mode = request.form['mode']
    
    files = request.files.to_dict(flat=False)['file']
    file_names = []
    for _file in files:
        filename_save = filemngr.get_file_save_location(allocate_job_key,_file.filename)
        file_names.append([{"key":"origin","value":_file.filename}])
        match mode:
            case "consolidate" | "process":
                filemngr.save_file(allocate_job_key,job_identity_current_job_id,'chembl',_file,file_origin=FileManager.FILE_TYPE_WEB_REQUESTS)
            case "preprocess":
                filemngr.save_file(allocate_job_key,job_identity_current_job_id,'consolidate',_file,file_origin=FileManager.FILE_TYPE_WEB_REQUESTS)
            case "cluster":
                filemngr.save_file(allocate_job_key,job_identity_current_job_id,'processed',_file,file_origin=FileManager.FILE_TYPE_WEB_REQUESTS)
            #case "analyze":
            #    filemngr.save_file(allocate_job_key,job_identity_current_job_id,'clustered',_file,file_origin=FileManager.FILE_TYPE_WEB_REQUESTS)
        _file.close()
    
    job_identity = filemngr.open_manifest(allocate_job_key)
    job_identity_kwargs = dict()
    match mode:
        case "consolidate":
            job_identity_kwargs['user_append'] = file_names
        case "preprocess":
            job_identity_kwargs['mfp_radius'] = int(request.form['mfp-radius'])
            job_identity_kwargs['mfp_size']   = int(request.form['mfp-size'])
        case "cluster":
            job_identity_kwargs['cluster_count'] = int(request.form['cluster-k'])
        case "process":
            job_identity_kwargs['user_append']   = file_names
            job_identity_kwargs['mfp_radius']    = int(request.form['mfp-radius'])
            job_identity_kwargs['mfp_size']      = int(request.form['mfp-size'])
            job_identity_kwargs['cluster_count'] = int(request.form['cluster-k'])
        case "analyze":
            pass
            #analysis = Analysis(file_buffers[0])()
            #return render_template("analysis.html",fig_bar_stacked=analysis.get_bar_stacked_clusters())
    job_identity['jobs'].append({'job_id':job_identity_current_job_id,'status':'allocated','mode':mode,'has_analysis':False,'kwargs':job_identity_kwargs})
    filemngr.overwrite_manifest(allocate_job_key,job_identity)

    request.close()
    jobs.start_job_from_manifest(filemngr,allocate_job_key)
    response = make_response(redirect(url_for('serve_index')))
    cookie_expiration = datetime.now()
    cookie_expiration = cookie_expiration + timedelta(days=3)
    response.set_cookie(f'{allocate_job_key}',allocate_job_key,expires=cookie_expiration)
    return response

#all together 1134 LOC