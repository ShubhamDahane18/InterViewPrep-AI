import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]: %(message)s:'
)

project_name = "interviewPrepAI"

list_of_files = [
    ".github/workflows/.gitkeep",

    f"src/{project_name}/__init__.py",
    f"src/{project_name}/controller/__init__.py",
    f"src/{project_name}/controller/core.py",

    f"src/{project_name}/resume_parser/__init__.py",
    f"src/{project_name}/resume_parser/parser.py",

    f"src/{project_name}/interview/__init__.py",
    f"src/{project_name}/interview/orchestrator.py",

    f"src/{project_name}/ai_analysis/__init__.py",
    f"src/{project_name}/ai_analysis/feedback.py",

    f"src/{project_name}/data_access/__init__.py",
    f"src/{project_name}/data_access/question_bank.py",
    f"src/{project_name}/data_access/user_profile.py",
    f"src/{project_name}/data_access/session_history.py",

    f"src/{project_name}/apis/__init__.py",
    f"src/{project_name}/apis/tts.py",
    f"src/{project_name}/apis/stt.py",
    f"src/{project_name}/apis/analysis.py",

    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",

    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",

    "app.py",
    "main.py",
    "Dockerfile",
    "requirements.txt",
    "config/config.yaml",
    "params.yaml",
    "setup.py",
    "research/notebook.ipynb"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory {filedir} for the file {filename}")
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
        logging.info(f"Creating empty file {filename} at {filedir}")
    else:
        logging.info(f"File {filename} already exists at {filedir}, skipping creation.")
