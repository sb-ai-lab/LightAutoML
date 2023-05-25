import requests


def counter():
    try:
        response = requests.get("http://10.102.27.150:9000/", timeout=2)
    except :
        pass
