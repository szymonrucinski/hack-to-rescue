# make a test request to the app
import requests


query = "Wie kann ich eine Tabelle erstellen?"
response = requests.post("http://localhost:8000/ask-model/", json={"query": query})
print(response.json())

response = requests.post("http://localhost:8000/inference-time/", json={"query": query})
print(response.json())


def query_apochat(prompt):
    try:
        response = requests.post("http://localhost:8000/ask-model/", json={"query": prompt})
        return response.json()["answer"

    response = requests.post("http://localhost:8000/ask-model/", json={"query": prompt})
    print(response.json())

