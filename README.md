# Target Structural Clustering
## Description

## Instructions
### Self-deployment (Windows)
Clone this repository to some folder on your device. In that folder create a python virtual environment and install all dependecies as required in file "requirements.txt".
Create an empty directory called "appdata" within the cloned repository. The file structure should look like this:
```
/
├── venv
│   ├── Scripts
│   │   ├── python.exe
│   │   └── ...
│   └── ...
└── TSC
    ├── appdata
    ├── modules
    │   └── ...
    ├── pages
    │   └── ...
    ├── resources
    │   └── ...
    ├── templates
    │   └── ...
    ├── main.py
    ├── config.py
    └── ...
```
Ensure that a `config.py` constant named `APPDATA_ABSOLUTE_PATH` is in fact the absolute path of the appdata folder listed above.

To start the application, open a terminal in the root folder and execute the following command:
`.\venv\Scripts\python.exe -m flask --app .\TSC\main.py run`
This will, by default, start the application on 127.0.0.1:5000
