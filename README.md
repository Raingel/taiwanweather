# taiwanweather
taiwanweather
`import requests
twCode = requests.get('https://raw.githubusercontent.com/Raingel/taiwanweather/main/taiwanWeather.py')
with open('./lib/taiwanWeather.py', 'w', encoding='utf-8') as f:
    f.write(twCode.text)
from lib.taiwanWeather import taiwanWeather`
