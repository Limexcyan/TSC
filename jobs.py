from socket import SocketIO
from modules.consolidator import Consolidator
from modules.featurizer import ClusteringFingerprintFeaturizer
from modules.clustering import BisectingKMeansClustering
from uuid import uuid4
from queue import Queue
from threading import Thread, Event
from modules.analyzer import Analysis
import traceback
from flask_socketio import emit
from fileManager import FileManager
from pickle import load as pickle_load

class JobsManager():
    def __init__(self, queue_max, appcontext, filemngr, **kwargs):
        self.queue_max = queue_max
        self.jobs = dict()
        self.queue = Queue(maxsize=queue_max)
        self.consumer_thread = Thread(target = consume_jobs, args=(self,appcontext,filemngr,), daemon=True)
        self.__dict__.update(kwargs)
        self.consumer_thread.start()

    def allocate_known_job(self,job_id,manifest_token):
        job_token = uuid4().hex
        self.jobs[job_token] = {"status":"allocated","job_id":job_id,"manifest_token":manifest_token}
        return job_token

    def allocate_job(self,job_id):
        job_token = uuid4().hex
        self.jobs[job_token] = {"status":"allocated","job_id":job_id,"manifest_token":job_token}
        return job_token

    def start_job_from_manifest(self, filemngr, job_token):
        self.update_job_status(job_token,'queued',filemngr=filemngr)
        self.queue.put_nowait(job_token)
        return job_token

    def get_job(self, job_token):
        if job_token in self.jobs:
            return self.jobs[job_token]
        return None

    def update_job_status(self, job_token, status, filemngr=None):
        if job_token in self.jobs:
            self.jobs[job_token]["status"] = status
            if not (filemngr is None):
                manifest_token = self.jobs[job_token]["manifest_token"]
                manifest = filemngr.open_manifest(manifest_token)
                manifest['jobs'][self.get_job(job_token)['job_id']]['status'] = status
                filemngr.overwrite_manifest(manifest_token, manifest)

    def pop_job(self, job_token):
        if job_token in self.jobs:
            return self.jobs.pop(job_token)
        print(" * Attempted to pop job which does not exist - this will cause issues!")
        return None

    def get_job_status(self, job_token):
        if job_token in self.jobs:
            return self.jobs[job_token]["status"]
        return "invalid"

def consume_jobs(manager: JobsManager, files_manager: FileManager, appcontext):
    while True:
        next_job_token = manager.queue.get(block=True, timeout=None)
        print(f"[consumer thread] - job {next_job_token} processing")
        manager.update_job_status(next_job_token, "processing",filemngr=files_manager)
        manifest_token = manager.get_job(next_job_token)['manifest_token']
        manifest = files_manager.open_manifest(manifest_token)
        job_id = manager.get_job(next_job_token)['job_id']
        latest_job = manifest['jobs'][job_id]
        mode = latest_job['mode']
        job_kwargs = latest_job['kwargs']
        content = files_manager.get_content_by_job_id(manifest_token,job_id)
        job = Job(mode, content, **job_kwargs)
        try:
            job()
            res = job.get_result()
            result_file_type = None
            file_origin = FileManager.FILE_TYPE_CSV_RESULTS
            match mode:
                case "consolidate":
                    result_file_type = "consolidate"
                case "preprocess":
                    result_file_type = "processed"
                case "process" | "recluster" | "cluster":
                    result_file_type = "clustered"
                case "analyze":
                    result_file_type = "obj_analyze_graph"
                    file_origin = FileManager.FILE_TYPE_HTML
            if mode == "recluster":
                files_manager.overwrite_file(manifest_token,job_id,result_file_type,"process",res,file_origin=file_origin)
                resource = job.get_resource()
                files_manager.overwrite_file(manifest_token,job_id,"obj_cluster_pickle","clustering",res,file_origin=FileManager.FILE_TYPE_PICKLE_OUTPUT)
            else:
                files_manager.save_file(manifest_token,job_id,result_file_type,res,file_origin=file_origin)
            if mode == "process" or mode == "cluster":
                resource = job.get_resource()
                files_manager.save_file(manifest_token,job_id,"obj_cluster_pickle",resource,file_origin=FileManager.FILE_TYPE_PICKLE_OUTPUT)
        except Exception as e:
            print(f"[consumer thread] - ran into error processing job of identity: \n{latest_job}","details:",e,traceback.format_exc(),sep="\n")
            manager.update_job_status(next_job_token,"failed",filemngr=files_manager)
        else:
            manager.update_job_status(next_job_token,"complete",filemngr=files_manager)
        finally:
            manager.queue.task_done()
            print(f"[consumer thread] - job freed")

class Job():
    def __init__(self, mode, content, **kwargs):
        self.mode = mode
        self.content = content
        self.resource = None
        self.result = None
        self.__dict__.update(kwargs)

    def __call__(self):

        match self.mode:
            case "consolidate":
                self.resource = Consolidator()
                self.result = self.resource(self.content,user_append=self.__dict__["user_append"])
            case "preprocess":
                self.content = [self.content[0]]
                # content is always list of files
                # resource for this mode accepts single file
                self.resource = ClusteringFingerprintFeaturizer(morganfp_size=self.__dict__["mfp_size"], morganfp_radius=self.__dict__["mfp_radius"])
                self.result = self.resource(self.content[0])
            case "cluster":
                self.content = [self.content[0]]
                # content is always list of files
                # resource for this mode accepts single file
                self.resource = BisectingKMeansClustering(k=self.__dict__["cluster_count"])
                _, self.result = self.resource(self.content[0])
            case "process":
                self.resource = None
                whole_resource = Consolidator()
                partial_result = whole_resource(self.content,user_append=self.__dict__["user_append"])
                whole_resource = ClusteringFingerprintFeaturizer(morganfp_size=self.__dict__["mfp_size"], morganfp_radius=self.__dict__["mfp_radius"])
                partial_result = whole_resource(None,in_df=partial_result)
                self.resource = BisectingKMeansClustering(k=self.__dict__["cluster_count"])
                _, self.result = self.resource(None,dataset=partial_result)
            case "analyze":
                self.resource = Analysis(self.content[0])
                self.result = self.resource().get_bar_stacked_clusters()
            case "recluster":
                resource_file = open(self.content[-1],"rb")
                self.resource = pickle_load(resource_file)
                resource_file.close()
                appendto = self.content[-2]
                new_content = self.content[:-2]
                temp_resource = Consolidator()
                new_content = temp_resource(new_content,user_append=self.__dict__["user_append"])
                temp_resource = ClusteringFingerprintFeaturizer(morganfp_size=self.__dict__["mfp_size"], morganfp_radius=self.__dict__["mfp_radius"])
                new_content = temp_resource(None,in_df=new_content)
                _, self.result = self.resource.transform(None,dataset=new_content,appendto=appendto)
            case _:
                pass
        return self

    def get_result(self):
        return self.result

    def get_resource(self):
        return self.resource

    def get_mode(self):
        return self.mode
