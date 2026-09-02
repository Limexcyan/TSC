function _api_call(method, endpoint, body="{}"){
    var xhr = new XMLHttpRequest();
    return new Promise((resolve, reject) => {
        xhr.onload = function () {
            if (xhr.status > 400) {
                console.log("request failed");
            } else {
                if (xhr.responseText.length < 1)
                    resolve(JSON.parse("{}"));
                else
                    resolve(JSON.parse(xhr.responseText));
            }
        }
        xhr.open(method, endpoint, true);
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.send(body);
    })
}

function api_post_new_client_session(preferred_name=null){
    if (preferred_name === null) {
        return _api_call("POST", "/api/session");
    } else {
        return _api_call("POST", "/api/session", `{"name":"${preferred_name}"`);
    }
}

function api_delete_existing_session(uuid){
    return _api_call("DELETE", `/api/session/${uuid}`);
}

function client_sessions_build_session_list(){
    client_sessions_build_session_list_file()
    client_sessions_build_session_list_task()
}

function client_sessions_build_session_list_file(){
    let root = document.getElementById("file-session-list");
    _api_call("GET", "/api/session")
        .then(res => {
            res.sort(function (a,b) {
                return new Date(b[3]) - new Date(a[3]);
            });
            root.innerHTML = '<h2>Active sessions</h2>';
            res.forEach(function (session) {
                session_container = document.createElement("div");
                session_container.className = "div-hoverable unhoverable"
                session_name = document.createElement("h3");
                session_manage = document.createElement("button");
                session_delete = document.createElement("button");
                session_name.innerText = session[2];
                session_manage.innerText = "Manage";
                session_delete.innerText = "Delete";
                session_manage.onclick = function () { window.location.href=`/session/${session[0]}`; };
                session_delete.onclick = function () { api_delete_existing_session(session[0]).then(client_sessions_build_session_list_file);};
                session_delete.classList.add("btn-red") 
                session_container.appendChild(session_name)
                session_container.appendChild(session_manage)
                session_container.appendChild(session_delete)
                root.appendChild(session_container)
            });
        });
}

function client_sessions_build_session_list_task(){
    let root = document.getElementById("task-session-list");
    _api_call("GET", "/api/task")
        .then(res => {
            console.log(res)
            res.sort(function (a,b) {
                return new Date(b[5]) - new Date(a[5]);
            });
            root.innerHTML = '<h2>Tasks</h2>';
            res.forEach(function (task) {
                task_container = document.createElement("div");
                task_container.className = "div-hoverable unhoverable"
                task_name = document.createElement("h3");
                task_status = document.createElement("span");
                task_uuid = document.createElement("h5");
                task_delete = document.createElement("button");
                task_resubmit = document.createElement("button");
                task_view_result = document.createElement("button");
                task_status.innerText = task[6];
                task_uuid.innerText = task[5];
                task_delete.innerText = "Delete";
                task_resubmit.innerText = "Resubmit";
                task_view_result.innerText = "View results";
                task_delete.onclick = function () { _api_call("DELETE",`/api/task/${task[0]}`).then(client_sessions_build_session_list_task)};
                task_resubmit.onclick = function () { _api_call("PUT",`/api/task/${task[0]}`, '{"status":"16"}').then(client_sessions_build_session_list_task)};
                task_view_result.onclick = function () {window.location.href = `/task/${task[0]}`}
                task_delete.classList.add("btn-red")
                switch(task[4]) {
                    case 2:
                        task_status.className = "material-badge material-badge-info"
                        break
                    case 4:
                        task_status.className = "material-badge material-badge-success"
                        break
                    case 8:
                        task_status.className = "material-badge material-badge-error"
                        break
                    case 16:
                        task_status.className = "material-badge material-badge-warning"
                        break
                    default:
                        task_status.className = "material-badge material-badge-primary"
                }
                task_container.appendChild(task_status)
                task_container.appendChild(task_name)
                task_container.appendChild(task_uuid)
                task_container.appendChild(task_delete)
                task_container.appendChild(task_resubmit)
                if (JSON.parse(task[3]).hasOwnProperty("result")) {
                    task_name.innerText = JSON.parse(task[3])['result']['output-filename'];
                } else {
                    task_name.innerText = "[unnamed task]";
                }
                if (task[4] == 4) {
                    task_container.append(task_view_result);
                }
                root.appendChild(task_container)
            });
        });
}

function upload_button_wake_up(){
    let gathered = Array.from(document.querySelectorAll("div[data-selected='1']"))
    let files = gathered.map((x) => x.id)
    if (files.length >= 1) {
        document.getElementById("submit-task").disabled = false
    } else {
        document.getElementById("submit-task").disabled = false
    }
}

function _client_session_build_sdf_files(uuid, file) {
    let root = document.getElementById("session-list-files-sdf")
    let file_container = document.createElement("div");
    let id = file[0].replaceAll("\\","/").split("/",3).at(-1);
    file_container.id = id
    file_container.dataset.selected = 0
    file_name = document.createElement("h3");
    file_delete = document.createElement("button");
    file_name.innerText = id;
    file_delete.innerText = "Delete";
    file_delete.onclick = function () { _api_call("DELETE", `/api/file/${id}`, `{"session":"${uuid}"}`).then(() => {client_session_build_session_contents(uuid)})}
    file_delete.classList.add("btn-red")
    file_container.appendChild(file_name);
    file_container.appendChild(file_delete);
    
    file_container.classList.add("div-hoverable");
    file_container.onclick = function () {
        file_selected_status = document.getElementById(id)
        if (file_selected_status.dataset.selected > 0) {
            file_selected_status.dataset.selected = 0;
            file_container.className = "div-hoverable";
        } else {
            file_selected_status.dataset.selected = 1;
            file_container.className = "div-selected";
        }
        upload_button_wake_up()
    }
    
    root.appendChild(file_container)
}

function _client_session_build_csv_files(uuid, file) {
    root = document.getElementById("session-list-files-csv")
    let file_container = document.createElement("div");
    let id = self.crypto.randomUUID();
    file_container.id = id
    file_container.dataset.selected = 0
    file_container.className = "div-hoverable unhoverable"
    file_name = document.createElement("h3");
    file_download = document.createElement("button");
    associated_task_view = document.createElement("button");
    let fn = file[0].replaceAll("\\","/").split("/",3).at(-1);
    let tn = file[0].replaceAll("\\","/").split("/",3).at(-2);
    file_name.innerText = fn;
    file_download.innerText = "Download";
    associated_task_view.innerText = "View results";
    file_download.onclick = function () { window.open(`/api/file/task/${tn}/${fn}`) };
    associated_task_view.onclick = function () { window.location.href = `/task/${tn}` };
    file_container.appendChild(file_name);
    file_container.appendChild(file_download);
    file_container.appendChild(associated_task_view);


    root.appendChild(file_container)
}

function send_files_task_clustering(uuid, filelist){
    return _api_call("POST", `/api/task`, `{"session":"${uuid}","data":{"func":"webapp_processor","kwargs":{"config": {"input_files":${filelist}}}}}`)
}


function client_session_build_session_contents(uuid){
    _api_call("GET", `/api/session/file/${uuid}`)
        .then(res => {
            res.sort();
            root = document.getElementById("session-list-files-sdf");
            root.innerHTML = '';
            root = document.getElementById("session-list-files-csv");
            root.innerHTML = '';
            res.forEach(function (file) {
                if (file[1] == "sdf") {
                    _client_session_build_sdf_files(uuid, file)
                } else if (file[1] == "csv") {
                    _client_session_build_csv_files(uuid, file)
                }
            });
        }
        );
}

function client_session_build_tasks(uuid){
    let root = document.getElementById("session-list-tasks")
    _api_call("GET", `/api/task/session/${uuid}`)
    .then(res => {
        console.log(res)
        res.sort(function (a,b) {
            return new Date(b[5]) - new Date(a[5]);
        });
        root.innerHTML = '';
        res.forEach(function (task) {
                task_container = document.createElement("div");
                task_container.className = "div-hoverable unhoverable"
                task_name = document.createElement("h3");
                task_status = document.createElement("span");
                task_uuid = document.createElement("h5");
                task_delete = document.createElement("button");
                task_resubmit = document.createElement("button");
                task_view_result = document.createElement("button");
                task_status.innerText = task[6];
                task_uuid.innerText = task[5];
                task_delete.innerText = "Delete";
                task_resubmit.innerText = "Resubmit";
                task_view_result.innerText = "View results";
                task_delete.onclick = function () { _api_call("DELETE",`/api/task/${task[0]}`).then(() => {client_session_build_tasks(uuid); client_session_build_session_contents(uuid);})};
                task_resubmit.onclick = function () { _api_call("PUT",`/api/task/${task[0]}`, '{"status":"16"}').then(() => {client_session_build_tasks(uuid)})};
                task_view_result.onclick = function () {window.location.href = `/task/${task[0]}`}
                task_delete.classList.add("btn-red")
                switch(task[4]) {
                    case 2:
                        task_status.className = "material-badge material-badge-info"
                        break
                    case 4:
                        task_status.className = "material-badge material-badge-success"
                        break
                    case 8:
                        task_status.className = "material-badge material-badge-error"
                        break
                    case 16:
                        task_status.className = "material-badge material-badge-warning"
                        break
                    default:
                        task_status.className = "material-badge material-badge-primary"
                }
                task_container.appendChild(task_status)
                task_container.appendChild(task_name)
                task_container.appendChild(task_uuid)
                task_container.appendChild(task_delete)
                task_container.appendChild(task_resubmit)
                if (JSON.parse(task[3]).hasOwnProperty("result")) {
                    task_name.innerText = JSON.parse(task[3])['result']['output-filename'];
                } else {
                    task_name.innerText = "[unnamed task]";
                }
                if (task[4] == 4) {
                    task_container.append(task_view_result);
                }
                root.appendChild(task_container)
            });
    })
}