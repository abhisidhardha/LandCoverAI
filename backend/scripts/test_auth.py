import requests

BASE = "http://127.0.0.1:5000"

r1 = requests.post(BASE + "/api/register", json={"name": "Test", "email": "test@example.com", "password": "password"})
print("register", r1.status_code, r1.text)

r2 = requests.post(BASE + "/api/login", json={"email": "test@example.com", "password": "password"})
print("login", r2.status_code, r2.text)
