import os
import json
from io import StringIO
from datetime import datetime
import pickle
from uuid import uuid4
from copy import deepcopy

class FileManager():

    FILE_TYPE_WEB_REQUESTS  = "webrequest"
    FILE_TYPE_CSV_RESULTS   = "csv_result"
    FILE_TYPE_PICKLE_OUTPUT = "pickle_clustering_result"
    FILE_TYPE_HTML = "html_output"
    OBFUSCATE_ROOT = "/"

    def __init__(self, appdata_path):
        self.appdata_path = appdata_path
        self.manifest = [_dir for _dir in os.listdir(appdata_path) if os.path.isdir(os.path.join(appdata_path,_dir))]
        self.expect_manifest_version_major = 3
        self.expect_manifest_version_minor = 0
        print("[file manager] - manifest loaded")

    def __call__(self):
        pass

    def create_directory(self, key):
        if key in self.manifest:
            raise RuntimeError(f"[file manager] - {key} already in file manager manifest!")
        os.mkdir(os.path.join(self.appdata_path, key))
        self.manifest.append(key)

    def create_manifest(self, key):
        if not key in self.manifest:
            raise RuntimeError(f"[file manager] - {key} not in file manager manifest!")
        manifest_info = open(os.path.join(self.appdata_path, key, "manifest.info"), 'w')
        manifest_info.write(json.dumps({
            'request_key' : key,
            'date_modified' : f'{datetime.now()}',
            'manifest_version_major' : self.expect_manifest_version_major,
            'manifest_version_minor' : self.expect_manifest_version_minor,
            'files' : {
                'chembl' : [
                    # 'filename' : ...
                    # 'job_id' : ...
                ],
                'consolidate' : [
                    # 'filename' : ...
                    # 'job_id' : ...
                ],
                'processed' : [
                    # 'filename' : ...
                    # 'job_id' : ...
                ],
                'clustered' : [
                    # 'filename' : ...
                    # 'job_id' : ...
                ],
                'obj_cluster_pickle' : [
                    # 'filename' : ...
                    # 'job_id' : ...
                ],
                'obj_analyze_graph' : [
                    # 'filename' : ...
                    # 'job_id' : ...
                ]
            },
            'jobs' : [
                # 'job_id' : ...
                # 'status' : allocated / queued / processing / complete / failed / cancelled
                # 'mode' : consolidate / process / preprocess / cluster / analyze
                # 'has_analysis' : true / false
                # 'kwargs' : ...
            ]
        }))
        manifest_info.close()
    
    def open_manifest(self, key):
        if not key in self.manifest:
            raise RuntimeError(f'[file manager] - {key} not in file manager manifest!')
        manifest_info = open(os.path.join(self.appdata_path, key, "manifest.info"), 'r')
        manifest_json = json.loads(manifest_info.read())
        manifest_info.close()
        if manifest_json['manifest_version_major'] != self.expect_manifest_version_major:
            raise RuntimeError(f'{key} inner manifest is of version {manifest_json["manifest_version_major"]}.{manifest_json["manifest_version_minor"]}, expected version is {self.expect_manifest_version_major}.{self.expect_manifest_version_minor}')
        if manifest_json['manifest_version_minor'] != self.expect_manifest_version_minor:
            print(f'[file manager] - Warning: {key} inner manifest version minor is {manifest_json["manifest_version_minor"]}, expected {self.expect_manifest_version_minor}')
        return manifest_json

    def overwrite_manifest(self, key, manifest_json):
        if not key in self.manifest:
            raise RuntimeError(f'[file manager] - {key} not in file manager manifest!')
        manifest_json['date_modified'] = f'{datetime.now()}'
        manifest_info = open(os.path.join(self.appdata_path, key, "manifest.info"), 'w')
        manifest_info.write(json.dumps(manifest_json))
        manifest_info.close()

    def query_jobless_manifest(self):
        jobless = []
        for key in self.manifest:
            for job in self.open_manifest(key)['jobs']:
                if job['status'] == 'queued':
                    jobless.append({"_job":job,"_manifest":key})
        return jobless

    def query_processing_manifest(self):
        processing = []
        for key in self.manifest:
            for job in self.open_manifest(key)['jobs']:
                if job['status'] == 'processing':
                    processing.append({"_job":job,"_manifest":key})
        return processing

    def get_file_save_location(self, key, fn):
        if not key in self.manifest:
            raise RuntimeError(f'[file manager] - {key} not in file manager manifest!')
        return os.path.join(self.appdata_path, key, fn)

    def mutate_manifest_replicate_asset(self, key, asset_type, fn, new_job_id):
        manifest = self.open_manifest(key)
        asset = [f for f in manifest["files"][asset_type] if fn in f["filename"]]
        if len(asset) < 1:
            raise RuntimeError(f"[file manager] - no asset found for {key} {asset_type} {fn}")
        asset = asset[0]
        newasset = deepcopy(asset)
        newasset["job_id"] = new_job_id
        manifest["files"][asset_type].append(newasset)
        self.overwrite_manifest(key, manifest)

    def prepare_for_analysis(self, key, job_id):
        manifest = self.open_manifest(key)
        jobs = len(manifest["jobs"])
        relevant_files = manifest["files"]["clustered"]
        _files = [_fn for _fn in relevant_files if int(job_id) == int(_fn["job_id"])]
        for _f in _files:
            self.mutate_manifest_replicate_asset(key,"clustered",_f["filename"],jobs)
        manifest = self.open_manifest(key)
        manifest["jobs"][job_id]["has_analysis"] = True
        self.overwrite_manifest(key, manifest)
        return jobs
    
    def prepare_for_reclustering(self, key, job_id):
        manifest = self.open_manifest(key)
        jobs = len(manifest["jobs"])
        relevant_files = manifest["files"]["clustered"] + manifest["files"]["obj_cluster_pickle"]
        _files = [_fn for _fn in relevant_files if int(job_id) == int(_fn["job_id"])]
        for _f in _files:
            if _f["filename"].split(".")[-1] == "pckl":
                self.mutate_manifest_replicate_asset(key,"obj_cluster_pickle",_f["filename"],jobs)
            else:
                self.mutate_manifest_replicate_asset(key,"clustered",_f["filename"],jobs)
        return jobs

    def get_content_by_job_id(self, key, job_id):
        job_manifest = self.open_manifest(key)
        if job_id > len(job_manifest['jobs']):
            raise RuntimeError(f'[file manager] - job_id {job_id} out of bounds!')
        job_mode = job_manifest['jobs'][job_id]['mode']
        content = []
        match job_mode:
            case "consolidate" | "process":
                content = [obj['filename'].replace(self.OBFUSCATE_ROOT,self.appdata_path,1) for obj in job_manifest['files']['chembl'] if obj['job_id'] == job_id]
            case "preprocess":
                content = [obj['filename'].replace(self.OBFUSCATE_ROOT,self.appdata_path,1) for obj in job_manifest['files']['consolidate'] if obj['job_id'] == job_id]
            case "cluster":
                content = [obj['filename'].replace(self.OBFUSCATE_ROOT,self.appdata_path,1) for obj in job_manifest['files']['processed'] if obj['job_id'] == job_id]
            case "analyze":
                content = [obj['filename'].replace(self.OBFUSCATE_ROOT,self.appdata_path,1) for obj in job_manifest['files']['clustered'] if obj['job_id'] == job_id] + [obj['filename'].replace(self.OBFUSCATE_ROOT,self.appdata_path,1) for obj in job_manifest['files']['obj_cluster_pickle'] if obj['job_id'] == job_id]
            case "recluster":
                content = [obj['filename'].replace(self.OBFUSCATE_ROOT,self.appdata_path,1) for obj in job_manifest['files']['chembl'] if obj['job_id'] == job_id] + [obj['filename'].replace(self.OBFUSCATE_ROOT,self.appdata_path,1) for obj in job_manifest['files']['clustered'] if obj['job_id'] == job_id] + [obj['filename'].replace(self.OBFUSCATE_ROOT,self.appdata_path,1) for obj in job_manifest['files']['obj_cluster_pickle'] if obj['job_id'] == job_id]
        return content
    
    def get_files_by_job_id(self, key, job_id):
        job_manifest = self.open_manifest(key)
        if job_id > len(job_manifest['jobs']):
            raise RuntimeError(f'[file manager] - job_id {job_id} out of bounds!')
        files = [obj['filename'].replace(self.OBFUSCATE_ROOT,self.appdata_path,1) for obj in job_manifest['files'][file_type] if obj["job_id"] == job_id for file_type in ["chembl","consolidate","processed","clustered","obj_cluster_pickle"]]
        return files

    def overwrite_file(self, key, job_id, file_type, old_file_name, _file, file_origin=None):
        manifest_json = self.open_manifest(key)
        uid = str(uuid4())[:8]
        _old_manifest_file_metadata_index = [i for i in range(len(manifest_json["files"][file_type])) if manifest_json["files"][file_type][i]["job_id"] == job_id and old_file_name in manifest_json["files"][file_type][i]["filename"]][0]
        match file_origin:
            case FileManager.FILE_TYPE_WEB_REQUESTS:
                _file_save_location = self.get_file_save_location(key, f"{uid}_{_file.filename}")
                manifest_json["files"][file_type][_old_manifest_file_metadata_index] = {"filename":_file_save_location.replace(self.appdata_path,self.OBFUSCATE_ROOT,1),"job_id":job_id}
                _file.save(_file_save_location)
            case FileManager.FILE_TYPE_CSV_RESULTS:
                job_mode = manifest_json["jobs"][job_id]['mode']
                _file_save_location = self.get_file_save_location(key, f"{uid}_{job_mode}.csv")
                manifest_json["files"][file_type][_old_manifest_file_metadata_index] = {"filename":_file_save_location.replace(self.appdata_path,self.OBFUSCATE_ROOT,1),"job_id":job_id}
                _file.to_csv(_file_save_location)
            case FileManager.FILE_TYPE_PICKLE_OUTPUT:
                _file_save_location = self.get_file_save_location(key, f"{uid}_clustering.pckl")
                manifest_json["files"][file_type][_old_manifest_file_metadata_index] = {"filename":_file_save_location.replace(self.appdata_path,self.OBFUSCATE_ROOT,1),"job_id":job_id}
                _f = open(_file_save_location,'wb')
                pickle.dump(_file, _f)
                _f.close()
            case FileManager.FILE_TYPE_HTML:
                _file_save_location = self.get_file_save_location(key, f"{uid}_graph.html")
                manifest_json["files"][file_type][_old_manifest_file_metadata_index] = {"filename":_file_save_location.replace(self.appdata_path,self.OBFUSCATE_ROOT,1),"job_id":job_id}
                _f = open(_file_save_location,"w")
                _f.write(_file)
                _f.close()
        self.overwrite_manifest(key, manifest_json)

    def save_file(self, key, job_id, file_type, _file, file_origin=None):
        manifest_json = self.open_manifest(key)
        uid = str(uuid4())[:8]
        match file_origin:
            case FileManager.FILE_TYPE_WEB_REQUESTS:
                _file_save_location = self.get_file_save_location(key, f"{uid}_{_file.filename}")
                manifest_json["files"][file_type].append({"filename":_file_save_location.replace(self.appdata_path,self.OBFUSCATE_ROOT,1),"job_id":job_id})
                _file.save(_file_save_location)
            case FileManager.FILE_TYPE_CSV_RESULTS:
                job_mode = manifest_json["jobs"][job_id]['mode']
                _file_save_location = self.get_file_save_location(key, f"{uid}_{job_mode}.csv")
                manifest_json["files"][file_type].append({"filename":_file_save_location.replace(self.appdata_path,self.OBFUSCATE_ROOT,1),"job_id":job_id})
                _file.to_csv(_file_save_location)
            case FileManager.FILE_TYPE_PICKLE_OUTPUT:
                _file_save_location = self.get_file_save_location(key, f"{uid}_clustering.pckl")
                manifest_json["files"][file_type].append({"filename":_file_save_location.replace(self.appdata_path,self.OBFUSCATE_ROOT,1),"job_id":job_id})
                _f = open(_file_save_location,'wb')
                pickle.dump(_file, _f)
                _f.close()
            case FileManager.FILE_TYPE_HTML:
                _file_save_location = self.get_file_save_location(key, f"{uid}_graph.html")
                manifest_json["files"][file_type].append({"filename":_file_save_location.replace(self.appdata_path,self.OBFUSCATE_ROOT,1),"job_id":job_id})
                _f = open(_file_save_location,"w")
                _f.write(_file)
                _f.close()
        self.overwrite_manifest(key, manifest_json)

    def pretty_print_statuses(self, key):
        table = StringIO()
        manifest = self.open_manifest(key)
        jobs = len(manifest["jobs"])
        for job in range(jobs):
            job_obj = manifest["jobs"][job]
            files_obfuscated = []
            for file_type in ["chembl","consolidate","processed","clustered","obj_cluster_pickle","obj_analyze_graph"]:
                files_obfuscated += [{"fn":obj['filename'].replace(self.OBFUSCATE_ROOT,"",1).replace(key,"").replace("\\","").replace("/",""),"src":file_type} for obj in manifest['files'][file_type] if obj["job_id"] == job]
            table.write(f"<div><h2>Task ID: {job}</h2>")
            table.write(f'<p>Type: {job_obj["mode"]}</p>')
            table.write(f'<p>Outcome: {job_obj["status"]}</p>')
            table.write(f"<p>Relevant files:</p><div><table><tbody>")
            for _f in files_obfuscated:
                table.write(f'<tr><td>{_f["fn"]}</td><td><a href="/download/{key}/{_f["src"]}/{_f["fn"]}">Download</a></td></tr>')
            table.write("</tbody></table></div></div>")
            if((job_obj["mode"] == "process" or job_obj["mode"] == "cluster" or job_obj["mode"] == "recluster") and job_obj["status"] == "complete"):
                table.write(f'<p><a href="/recluster/{key}/{job}">Run clustering on this instance</a></p>')
                if(not job_obj["has_analysis"]):
                    table.write(f'<p><a href="/analyze/{key}/{job}">Run analysis</a></p>')
            if(job_obj["mode"] == "analyze" and job_obj["status"] == "complete"):
                table.write(f'<p><a href="/analyze/{key}/{job}">Show analysis</a></p>')
        table_str = table.getvalue()
        table.close()
        return table_str

    def get_kwargs_for_processing(self, key, job_id):
        manifest = self.open_manifest(key)
        job_kwargs = manifest["jobs"][job_id]["kwargs"]
        return job_kwargs["mfp_radius"], job_kwargs["mfp_size"], job_kwargs["cluster_count"]
