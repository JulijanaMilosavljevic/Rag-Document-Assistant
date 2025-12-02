import threading
import time
import requests
from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Backend alive"

def ping():
    url = "https://rag-document-assistant-gt.onrender.com"
    while True:
        try:
            print("Pinging backend...")
            requests.get(url, timeout=10)
        except Exception as e:
            print("Ping error:", e)
        time.sleep(300)  # svakih 5 minuta

def start_ping():
    t = threading.Thread(target=ping)
    t.daemon = True
    t.start()

if __name__ == "__main__":
    start_ping()
    app.run(host="0.0.0.0", port=10000)
