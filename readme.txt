virtualenv .env --python=python3.11
source .env/bin/activate
deactivate

pip freeze > requirements.txt