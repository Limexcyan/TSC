function factory_api_delete(uuid, filename=null) {
    request = `/api/session/${uuid}`;
    _onload = function () {};
    if (filename === null) {
        _onload = function () { };
    } else {
        request = `/api/session/${uuid}/${filename}`;
        _onload = function () { window.location.reload(); };
    }
    return function () {
        var xhr = new XMLHttpRequest();
        xhr.open("DELETE", request, true); // may be async
        xhr.send("{}");
        xhr.onload = _onload;
    }
}

function factory_redirect_to_session(uuid) {
    return function () { window.location.href = `/session/${uuid}`; }
}

function fetch_session_files(uuid) {
    var xhr = new XMLHttpRequest();
    xhr.open("GET", `/api/session/${uuid}`, false) // no async to show files in order
    xhr.send("{}")
    return xhr
}

function fetch_session(top=10, skip=0){
    var xhr = new XMLHttpRequest();
    xhr.open("GET", `/api/session`, true)
    xhr.send(`{"top":${top},"skip":${skip}}`)
    return xhr
}

function create_button(inner_text, onclick) {
    btn = document.createElement("button")
    btn.innerText = inner_text
    btn.onclick = onclick
}

function create_listing_session(session_details) {
    anchor = document.getElementById("session-listing-root");
    session_listing = document.createElement("div");
    session_name = document.createElement("p");
    session_name.innerText = session_details.session_alias;
    session_listing.appendChild(session_name);
    session_buttons = document.createElement("div");
    session_button_delete = create_button("Delete session", factory_api_delete(session_details.session_id));
    session_button_manage_files = create_button("Manage files", factory_redirect_to_session(session_details.session_id));
    session_buttons.appendChild(session_button_delete);
    session_buttons.appendChild(session_button_manage_files);

    
}


function delete_factory(uuid) {
    return function () {
        var xhr = new XMLHttpRequest();
        xhr.open("DELETE", "/api/session/" + uuid, true);
        xhr.send("{}");
        xhr.onload = function () {
            build_list();
        };
    }
}

function delete_file_factory(uuid, filename) {
    return function () {
        var xhr = new XMLHttpRequest();
        xhr.open("DELETE", "/api/session/" + uuid + "/" + filename, true);
        xhr.send("{}");
        xhr.onload = function () {
            window.location.reload();
        };
    }
}

function get_factory(uuid) {
    return function () {
        location.href = '/session/' + uuid;
    }
}

function fetch_data(uuid) {
    var xhr = new XMLHttpRequest();
    xhr.open("GET", "/api/session/" + uuid, true);
    xhr.send("{}");
    xhr.onload = function() {
        response = JSON.parse(xhr.response);
        list_root = document.getElementById("list-root");
        let info = document.createElement("li");
        if (!!response.session_alias){
            info.innerText = response.session_alias;
        } else {
            info.innerText = response.session_id;
        }
        list_root.appendChild(info);
        let items = document.createElement("ul");
        list_root.appendChild(items);
        if (response.session_contents.length == 0) {
            let warn = document.createElement("li");
            warn.innerText = "No files";
            items.appendChild(warn);
        }
        for(let i = 0; i < response.session_contents.length; ++i) {
            let li = document.createElement("li");
            li.innerText = response.session_contents[i];
            items.appendChild(li);
        }
        let manage_btn = document.createElement("button");
        manage_btn.innerText = "manage session";
        manage_btn.onclick = get_factory(uuid);
        list_root.appendChild(manage_btn);
        let delete_btn = document.createElement("button");
        delete_btn.innerText = "delete session";
        delete_btn.onclick = delete_factory(uuid);
        list_root.appendChild(delete_btn);
    }
}
function build_list(skip=0, top=10) {
    if (!!document.getElementById("list-root")) {document.getElementById("list-root").remove()}
    var xhr = new XMLHttpRequest();
    xhr.open("GET", "/api/session", true);
    xhr.send("{}");
    xhr.onload = function() {
        response = JSON.parse(xhr.response);
        list_root = document.getElementById("session-list");
        if (response.length == 0) {
            let info = document.createElement("p");
            info.innerText = "No items to display";
            info.id = "list-root"
            list_root.appendChild(info);
        } else {
            let list = document.createElement("ul")
            list.id = "list-root";
            list_root.appendChild(list);
            response.slice(skip,skip+top).map(function (x) {fetch_data(x)});
        }
    }
}
function build_page(uuid) {
    var xhr = new XMLHttpRequest();
    xhr.open("GET", "/api/session/" + uuid, true);
    xhr.send("{}");
    xhr.onload = function() {
        response = JSON.parse(xhr.response);
        site_title = document.getElementById("session-title");
        site_title.innerText = response.session_alias;
        site_files = document.getElementById("session-files");
        if (response.session_contents.length == 0) {
            site_files.innerText = "No files in this session. Upload files"
        } else {
            for(let i = 0; i < response.session_contents.length; ++i) {
                let li = document.createElement("li");
                filename = response.session.session_contents[i];
                li.innerText = filename;
                site_files.appendChild(li);
                let delete_btn = document.createElement("button");
                delete_btn.innerText = "delete file";
                delete_btn.onclick = delete_file_factory(uuid, filename);
                site_files.appendChild(delete_btn);
            }
        }
    }
}

function build_file_item(uuid) {

}