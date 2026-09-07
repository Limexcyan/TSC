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
    let x = document.createElement(element);
    if(id !== undefined) { x.id = id; }
    if(innerText !== undefined) {x.innerText = innerText; }
    if(click !== undefined) { x.onclick = click; }
    if(className !== undefined) { x.className = className; }
    if(dataset !== undefined) { Object.keys(x).forEach(k => {x.dataset[k] = x[k]});}
    parentNode.appendChild(x);
    return x;
}

function _create_badge(parentNode, status, statusname, id){
    let task_status = "material-badge";
    switch(status) {
        case 2:
            task_status = "material-badge material-badge-info";
            break;
        case 4:
            task_status = "material-badge material-badge-success";
            break;
        case 8:
            task_status = "material-badge material-badge-error";
            break;
        case 16:
            task_status = "material-badge material-badge-warning";
            break;
        case 64:
            task_status = "material-badge material-badge-crashed";
            break;
        default:
            task_status = "material-badge";
    }
    _create_element(parentNode, "span", {id:`${id}-badge`,className:task_status,innerText:statusname});
}
function _update_badge(status, statusname, id){
    parent = document.getElementById(`${id}`);
    if (parent === null) {
        client_sessions_build_session_list_task();
        return;
    }
    btn_d = document.getElementById(`${id}-d`)
    btn_r = document.getElementById(`${id}-r`)
    btn_v = document.getElementById(`${id}-v`)
    btn_d?.remove()
    btn_r?.remove()
    btn_v?.remove()
    let task_status = "material-badge material-badge-info";
    switch(status) {
        case 2: // running
            task_status = "material-badge material-badge-info";
            break
        case 4: // success
            task_status = "material-badge material-badge-success";
            _create_element(parent, "button", {id:`${id}-d`,innerText:"Delete",click:(() => { _api_call("DELETE",`/api/task/${id}`).then(() => {client_sessions_build_session_list_task()})}),className:"btn-red"});
            _create_element(parent, "button", {id:`${id}-r`,innerText:"Resubmit",click:(() => { _api_call("PUT",`/api/task/${id}`, '{"status":"16"}').then(() => {client_sessions_build_session_list_task()})})});
            _create_element(parent, "button", {id:`${id}-v`,innerText:"View",click:(() => {window.location.href = `/task/${id}`})});
            break
        case 8: // failure
            task_status = "material-badge material-badge-error";
            _create_element(parent, "button", {id:`${id}-d`,innerText:"Delete",click:(() => { _api_call("DELETE",`/api/task/${id}`).then(() => {client_sessions_build_session_list_task()})}),className:"btn-red"});
            _create_element(parent, "button", {id:`${id}-r`,innerText:"Resubmit",click:(() => { _api_call("PUT",`/api/task/${id}`, '{"status":"16"}').then(() => {client_sessions_build_session_list_task()})})});
            _create_element(parent, "button", {id:`${id}-v`,innerText:"View",click:(() => {window.location.href = `/task/${id}`})});
            break
        case 16: // enqueued
            task_status = "material-badge material-badge-warning";
            _create_element(parent, "button", {id:`${id}-d`,innerText:"Delete",click:(() => { _api_call("DELETE",`/api/task/${id}`).then(() => {client_sessions_build_session_list_task()})}),className:"btn-red"});
            break
        case 64: // crashed
            task_status = "material-badge material-badge-crashed";
            _create_element(parent, "button", {id:`${id}-d`,innerText:"Delete",click:(() => { _api_call("DELETE",`/api/task/${id}`).then(() => {client_sessions_build_session_list_task()})}),className:"btn-red"});
            _create_element(parent, "button", {id:`${id}-r`,innerText:"Resubmit",click:(() => { _api_call("PUT",`/api/task/${id}`, '{"status":"16"}').then(() => {client_sessions_build_session_list_task()})})});
            break;
        default:
            task_status = "material-badge";
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

async function client_sessions_build_session_list(){
    webapp_build_session_list();
    client_sessions_build_session_list_task();
}

function _navigate_to_session(uuid){window.location.href = `/session/${uuid}`}

function webapp_build_session_list(){
    let root = document.getElementById("file-session-list");
    _api_call("GET", "/api/session")
        .then(res => {
            res.sort(function (a,b) {
                return new Date(b[3]) - new Date(a[3]);
            });
            root.innerHTML = '<h2>Active sessions</h2><h5>You can rename them by clicking on their name.</h5>';
            res.forEach(function (session) {
                let session_container = _create_element(root, "div", {id:session[0],className:"div-hoverable unhoverable"})

                // title
                title = _create_element(session_container, "h3", {innerText:session[2],className:"editable-header"});
                _register_header_editable(title, session[0]);
                // manage
                _create_element(session_container, "button", {innerText:"Manage", click:(() => {_navigate_to_session(session[0])})});
                // delete
                _create_element(session_container, "button", {innerText:"Delete", click:(() => { api_delete_existing_session(session[0]).then(() => {webapp_build_session_list()});}), className:"btn-red"});
            });
        });
}

async function client_sessions_build_session_list_task(uuid=null){
    let endpoint = "/api/task"
    if (uuid !== null) {
        endpoint = `/api/task/session/${uuid}`
    }
    let root = document.getElementById("task-session-list");
    let relevant_session = await _api_call("GET", "/api/session").then(res => {let r = {};res.map((x) => (r[`${x[0]}`]= x[2])); return r});
    await _api_call("GET", endpoint)
        .then(res => {
            res.sort(function (a,b) {
                return new Date(b[5]) - new Date(a[5]);
            });
            root.innerHTML = '<h2>Tasks</h2>';
            res.forEach(function (task) {
                let task_container = _create_element(root, "div", {id:task[0],className:"div-hoverable unhoverable"})

                let task_name = "[unnamed task]";
                if (JSON.parse(task[3]).hasOwnProperty("name")) {task_name = JSON.parse(task[3])['name'];}

                // name
                _create_element(task_container, "h3", {innerText:task_name});
                // badge
                _create_badge(task_container, task[4], task[6], task[0]);
                // time
                _create_element(task_container, "h5", {innerText:Date(task[5]).toString().split(" GMT")[0]});

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
        document.getElementById("submit-task").disabled = true
    }
}

function _file_container_select(id) {
    file_selected_status = document.getElementById(id)
    if (file_selected_status.dataset.selected > 0) {
        file_selected_status.dataset.selected = 0;
        file_selected_status.className = "div-hoverable";
    } else {
        file_selected_status.dataset.selected = 1;
        file_selected_status.className = "div-selected";
    }
    upload_button_wake_up()
}

function _client_session_build_sdf_files(uuid, file) {
    let root = document.getElementById("session-list-files-sdf")
    let id = file[0].replaceAll("\\","/").split("/",3).at(-1);

    let file_container = _create_element(root, "div", {id:id,dataset:{selected:0},className:"div-hoverable",click:(() => _file_container_select(id))})

    // name
    _create_element(file_container, "h3", {innerText:id})
    // delete
    _create_element(file_container, "button", {innerText:"Delete",click:(() => {_api_call("DELETE", `/api/file/${id}`, `{"session":"${uuid}"}`).then(() => {client_session_build_session_contents(uuid)})}),className:"btn-red"})
}

function _client_session_build_csv_files(uuid, file) {
    let root = document.getElementById("session-list-files-csv")
    let id = self.crypto.randomUUID();
    let fn = file[0].replaceAll("\\","/").split("/",3).at(-1);
    let tn = file[0].replaceAll("\\","/").split("/",3).at(-2);

    let file_container = _create_element(root, "div", {id:id,dataset:{selected:0},className:"div-hoverable unhoverable"})

    // name
    _create_element(file_container, "h3", {innerText:fn})
    // download
    _create_element(file_container, "button", {innerText:"Download",click:(() => {window.open(`/api/file/task/${tn}/${fn}`)})})
    // view
    _create_element(file_container, "button", {innerText:"View analysis",click:(() => {window.location.href = `/task/${tn}`})})

}


function _client_session_build_sdf_files(root_id, uuid, file, ignore_interact=false, selected=0) {
    let root = document.getElementById(root_id)
    let id = file[0].replaceAll("\\","/").split("/",3).at(-1);

    let file_container = _create_element(root, "div", {id:id,dataset:{selected:selected},className:"div-hoverable",click:(() => _file_container_select(id))})
    _file_container_select(id);
    _file_container_select(id);

    // name
    _create_element(file_container, "h3", {innerText:id})
    // delete
    if (!ignore_interact)
    _create_element(file_container, "button", {innerText:"Delete",click:(() => {_api_call("DELETE", `/api/file/${id}`, `{"session":"${uuid}"}`).then(() => {client_session_build_session_contents(uuid)})}),className:"btn-red"})
}

function send_files_task_clustering(uuid, filelist, mfp_r, mfp_c, c_c, c_s=null){
    kwargs = `{"session":"${uuid}","data":{"func":"webapp_processor","kwargs":{"config": {"input_files":${filelist}, "featurizer_config":{"mfp_radius":${mfp_r},"mfp_size":${mfp_c}},"clustering_config":{"cluster_count":${c_c}}}}}}`;
    if (c_s !== null) { kwargs = `{"session":"${uuid}","data":{"func":"webapp_processor","kwargs":{"config": {"input_files":${filelist}, "featurizer_config":{"mfp_radius":${mfp_r},"mfp_size":${mfp_c}},"clustering_config":{"cluster_count":${c_c},"seed":${c_s}}}}}}`}
    return _api_call("POST", `/api/task`, kwargs)
}


function resend_files_task_clustering(uuid, filelist, old_data, task_id){
    let filelist_array = Array.from(filelist);
    if (filelist.length < 2) {
        input_files = `"input_files":["${filelist_array}"]`
    } else {
        input_files = `"input_files":${filelist_array}`
    }
    let old_config = old_data.kwargs.config;
    kwargs = `{"session":"${uuid}","data":{"func":"webapp_reprocessor","kwargs":{"config": {${input_files}, "resource_path":"${task_id}", "featurizer_config":{"mfp_radius":${old_config.featurizer_config.mfp_radius},"mfp_size":${old_config.featurizer_config.mfp_size}},"clustering_config":{"cluster_count":${old_config.clustering_config.cluster_count},"seed":${old_config.clustering_config.seed}}}}}}`;
    return _api_call("POST", `/api/task`, kwargs)
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
                    _client_session_build_sdf_files("session-list-files-sdf", uuid, file)
                } else if (file[1] == "csv") {
                    _client_session_build_csv_files(uuid, file)
                }
            });
        }
        );
}

function client_task_build_session_sdf_files(uuid, files){
    _api_call("GET", `/api/session/file/${uuid}`)
        .then(res => {
            res.sort();
            root = document.getElementById("task-list-files-sdf");
            root.innerHTML = '';
            res.forEach(function (file) {
                if (file[1] == "sdf") {
                    let id = file[0].replaceAll("\\","/").split("/",3).at(-1);
                    console.log(id, files, files.includes(id))
                    if (files.includes(id)){
                        //_client_session_build_sdf_files("task-list-files-sdf", uuid, file, true, 1)
                    } else {
                        _client_session_build_sdf_files("task-list-files-sdf", uuid, file, true, 0)
                    }
                    
                }
            });
        }
        );
}



function _register_header_editable(header, id){

    header.addEventListener("click", (e) => {
        // Prevent creating another input if one already exists
        e.stopPropagation();
        if (header.querySelector("input")) return;

        const currentText = header.textContent.trim();

        const input = document.createElement("input");
        input.type = "text";
        input.value = currentText;
        input.setAttribute("aria-label", "Edit heading");

        header.textContent = "";
        header.appendChild(input);

        input.focus();
        input.select();

        input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            const newText = input.value.trim() || currentText;

            header.textContent = newText;

            // Custom event emitted when Enter is pressed
            header.dispatchEvent(
                new CustomEvent("header-submit", {
                    bubbles: true,
                    detail: {
                        id: id,
                        value: newText
                    }
                })
            );
        }

        if (event.key === "Escape") {
            header.textContent = currentText;
        }
        });

        input.addEventListener("blur", () => {
            if (header.contains(input)) {
            header.textContent = input.value.trim() || currentText;
            }
        });
        });

        // Listen for the emitted event
        header.addEventListener("header-submit", (event) => {
            _api_call("PUT", `/api/session/${event.detail.id}`, `{"name":"${event.detail.value}"}`)
    });
}

function load_task_data(uuid){
    let root = document.getElementById("task-data");
    return _api_call("GET", `/api/task/${uuid}`)
        .then((res) => {
            let files = res.data.kwargs.config.input_files
            root.innerText = JSON.stringify(res.data);
            document.getElementById("mfp-size").value = res.data.kwargs.config.featurizer_config.mfp_size;
            document.getElementById("mfp-radius").value = res.data.kwargs.config.featurizer_config.mfp_radius;
            document.getElementById("clu-count").value = res.data.kwargs.config.clustering_config.cluster_count;
            document.getElementById("clu-seed").value = res.data.kwargs.config.clustering_config.seed;
            
			document.getElementById("mfp-size-value").textContent = res.data.kwargs.config.featurizer_config.mfp_size;
			document.getElementById("mfp-radius-value").textContent = res.data.kwargs.config.featurizer_config.mfp_radius;
            return files;
        })
}