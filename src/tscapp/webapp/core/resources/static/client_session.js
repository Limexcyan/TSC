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
    root = document.getElementById("session-list");
    _api_call("GET", "/api/session")
        .then(res => {
            res.sort(function (a,b) {
                return new Date(b[3]) - new Date(a[3]);
            });
            root.innerHTML = '';
            res.forEach(function (session) {
                session_container = document.createElement("div");
                session_name = document.createElement("h2");
                session_manage = document.createElement("button");
                session_delete = document.createElement("button");
                session_name.innerText = session[2];
                session_manage.innerText = "Manage";
                session_delete.innerText = "Delete";
                session_manage.onclick = function () { window.location.href=`/session/${session[0]}`; };
                session_delete.onclick = function () { api_delete_existing_session(session[0]).then(client_sessions_build_session_list);};
                session_container.appendChild(session_name)
                session_container.appendChild(session_manage)
                session_container.appendChild(session_delete)
                root.appendChild(session_container)
            });
        });
}

function client_session_build_session_contents(uuid){
    root = document.getElementById("session-list");
    _api_call("GET", `/api/session/file/${uuid}`)
        .then(res => {
            console.log(res)
            res.sort();
            root.innerHTML = '';
            res.forEach(function (file) {
                file_container = document.createElement("div");
                file_name = document.createElement("h3");
                file_delete = document.createElement("button");
                file_name.innerText = file[0].replaceAll("\\","/").split("/",3).at(-1);;
                file_delete.innerText = "Delete"
                file_delete.onclick = function () { _api_call("DELETE", `/api/file/${file[0].replaceAll("\\","/").split("/",3).at(-1)}`, `{"session":"${uuid}"}`).then(() => {client_session_build_session_contents(uuid)})}
                file_container.appendChild(file_name)
                file_container.appendChild(file_delete)
                root.appendChild(file_container)
            });
        }
        );
}