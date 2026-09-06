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

function _create_element(parentNode, element, {id, innerText, click, className, dataset}){
    let x = document.createElement(element)
    if(id !== null) { x.id = id }
    if(innerText !== null) {x.innerText = innerText }
    if(click !== null) { x.onclick = click }
    if(className !== null) { x.className = className }
    if(dataset !== null) { Object.keys(x).forEach(k => {x.dataset[k] = x[k]})}
    parentNode.appendChild(x)
}

function _create_badge(parentNode, status, statusname, id){
    let task_status = "material-badge"
    switch(status) {
        case 2:
            task_status = "material-badge material-badge-info"
            break
        case 4:
            task_status = "material-badge material-badge-success"
            break
        case 8:
            task_status = "material-badge material-badge-error"
            break
        case 16:
            task_status = "material-badge material-badge-warning"
            break
        default:
            task_status = "material-badge"
    }
    _create_element(parentNode, "span", {id:`${id}-badge`,className:task_status,innerText:statusname})
}
function _update_badge(status, statusname, id){
    parent = document.getElementById(`${id}`)
    if (parent === null) {
        client_sessions_build_session_list_task()
        return
    }
    let task_status = "material-badge material-badge-info"
    switch(status) {
        case 2: // running
            task_status = "material-badge material-badge-info"
            document.getElementById(`${id}-d`).remove()
            document.getElementById(`${id}-r`).remove()
            document.getElementById(`${id}-v`).remove()
            break
        case 4: // success
            task_status = "material-badge material-badge-success"
            _create_element(parent, "button", {id:`${id}-d`,innerText:"Delete",click:(() => { _api_call("DELETE",`/api/task/${id}`).then(client_sessions_build_session_list_task)}),className:"btn-red"});
            _create_element(parent, "button", {id:`${id}-r`,innerText:"Resubmit",click:(() => { _api_call("PUT",`/api/task/${id}`, '{"status":"16"}').then(client_sessions_build_session_list_task)})});
            _create_element(parent, "button", {id:`${id}-v`,innerText:"View",click:(() => {window.location.href = `/task/${id}`})})
            break
        case 8: // failure
            task_status = "material-badge material-badge-error"
            _create_element(parent, "button", {id:`${id}-d`,innerText:"Delete",click:(() => { _api_call("DELETE",`/api/task/${id}`).then(client_sessions_build_session_list_task)}),className:"btn-red"});
            _create_element(parent, "button", {id:`${id}-r`,innerText:"Resubmit",click:(() => { _api_call("PUT",`/api/task/${id}`, '{"status":"16"}').then(client_sessions_build_session_list_task)})});
            _create_element(parent, "button", {id:`${id}-v`,innerText:"View",click:(() => {window.location.href = `/task/${id}`})})  
            break
        case 16: // enqueued
            task_status = "material-badge material-badge-warning"
            _create_element(parent, "button", {id:`${id}-d`,innerText:"Delete",click:(() => { _api_call("DELETE",`/api/task/${id}`).then(client_sessions_build_session_list_task)}),className:"btn-red"});
            document.getElementById(`${id}-r`).remove()
            document.getElementById(`${id}-v`).remove()
            break
        default:
            task_status = "material-badge"
    }
    badge = document.getElementById(`${id}-badge`);
    if (badge !== null) {
        badge.className = task_status;
        badge.innerText = statusname;
    }
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
    webapp_build_session_list()
    client_sessions_build_session_list_task()
}

function _navigate_to_session(uuid){window.location.href = `/session/${uuid}`}

function webapp_build_session_list(){
    let root = document.getElementById("file-session-list");
    _api_call("GET", "/api/session")
        .then(res => {
            res.sort(function (a,b) {
                return new Date(b[3]) - new Date(a[3]);
            });
            root.innerHTML = '<h2>Active sessions</h2>';
            res.forEach(function (session) {
                let session_container = document.createElement("div");
                session_container.id = session[0];
                session_container.className = "div-hoverable unhoverable";
                session_container.onclick = (() => {_navigate_to_session(session[0])});
                // title
                _create_element(session_container, "h3", {innerText:session[2]});
                // manage
                _create_element(session_container, "button", {innerText:"Manage", click:(() => {_navigate_to_session(session[0])})});
                // delete
                _create_element(session_container, "button", {innerText:"Delete", click:function () { api_delete_existing_session(session[0]).then(webapp_build_session_list);}, className:"btn-red"});
                root.appendChild(session_container);
            });
        });
}

async function client_sessions_build_session_list_task(){
    let root = document.getElementById("task-session-list");
    let relevant_session = await _api_call("GET", "/api/session").then(res => {let r = {};res.map((x) => (r[`${x[0]}`]= x[2])); return r});
    console.log(relevant_session)
    await _api_call("GET", "/api/task")
        .then(res => {
            console.log(res)
            res.sort(function (a,b) {
                return new Date(b[5]) - new Date(a[5]);
            });
            root.innerHTML = '<h2>Tasks</h2>';
            res.forEach(function (task) {
                task_container = document.createElement("div");
                task_container.id = task[0];
                task_container.className = "div-hoverable unhoverable"

                let task_name = "[unnamed task]";
                if (JSON.parse(task[3]).hasOwnProperty("result")) {task_name = JSON.parse(task[3])['result']['output-filename'];}

                _create_element(task_container, "h3", {innerText:task_name});
                _create_badge(task_container, task[4], task[6], task[0]);
                _create_element(task_container, "h5", {innerText:Date(task[5]).toString().split(" GMT")[0]});
                // _create_element(task_container, "button", {id:`${task[0]}-d`,innerText:"Delete",click:(() => { _api_call("DELETE",`/api/task/${task[0]}`).then(client_sessions_build_session_list_task)}),className:"btn-red"});
                // _create_element(task_container, "button", {id:`${task[0]}-r`,innerText:"Resubmit",click:(() => { _api_call("PUT",`/api/task/${task[0]}`, '{"status":"16"}').then(client_sessions_build_session_list_task)})});
                // _create_element(task_container, "button", {id:`${task[0]}-v`,innerText:"View",click:(() => {window.location.href = `/task/${task[0]}`})})

                root.appendChild(task_container);

                // run update badge to refresh button visibility
                _update_badge(task[4],task[6],task[0])
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
