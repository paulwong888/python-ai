install python:
    sudo yum install python3.12
    sudo yum install python3.12-devel

install virtual env:
    for mac:
        virtualenv .venv --python=python3.12.4
        source .venv/bin/activate
        deactivate

    or

    python3.12 -m venv .venv

    or

    use vscode to create.

install packages:
    pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/ --no-cache-dir
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

pip install python-dotenv

'Droid Sans Mono', 'monospace', monospace

anaconda:
    conda env export > environment.yml
    conda create --name first-ai python=3.12
    conda env create -f environment.yml --name target_env_name
    conda env update -f environment.yml
    conda env update -n target_env_name -f environment.yml
    conda install langchain -c conda-forge