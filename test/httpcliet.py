import requests

url="http://baidu.com"

resp = requests.get(url=url)
print(resp.status_code)
print(resp.text)