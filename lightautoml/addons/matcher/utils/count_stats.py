import requests


def counter():
    try:
        requests.get("10.102.27.150:9000/", timeout=3)
    except:
        pass

    return
