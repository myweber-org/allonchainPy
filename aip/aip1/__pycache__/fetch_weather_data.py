
import requests
import json

def get_weather(api_key, city_name):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['cod'] != 200:
            return None
        
        weather_info = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed']
        }
        
        return weather_info
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(weather_data):
    if not weather_data:
        print("Unable to retrieve weather information.")
        return
    
    print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels like: {weather_data['feels_like']}°C")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Conditions: {weather_data['description'].title()}")
    print(f"Wind Speed: {weather_data['wind_speed']} m/s")

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    CITY = "London"
    
    weather = get_weather(API_KEY, CITY)
    display_weather(weather)
import requests
import json
from datetime import datetime
import logging

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        logging.basicConfig(level=logging.INFO)
        
    def get_current_weather(self, city_name, country_code=None):
        query = city_name
        if country_code:
            query = f"{city_name},{country_code}"
            
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/weather",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return self._parse_weather_data(response.json())
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch weather data: {e}")
            return None
    
    def _parse_weather_data(self, raw_data):
        if not raw_data or 'main' not in raw_data:
            return None
            
        return {
            'city': raw_data.get('name'),
            'temperature': raw_data['main'].get('temp'),
            'feels_like': raw_data['main'].get('feels_like'),
            'humidity': raw_data['main'].get('humidity'),
            'pressure': raw_data['main'].get('pressure'),
            'weather': raw_data['weather'][0].get('description') if raw_data.get('weather') else None,
            'wind_speed': raw_data.get('wind', {}).get('speed'),
            'timestamp': datetime.fromtimestamp(raw_data.get('dt')).isoformat() if raw_data.get('dt') else None
        }
    
    def get_forecast(self, city_name, days=5):
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': 'metric',
            'cnt': days
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/forecast",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return self._parse_forecast_data(response.json())
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch forecast: {e}")
            return None
    
    def _parse_forecast_data(self, raw_data):
        if not raw_data or 'list' not in raw_data:
            return None
            
        forecasts = []
        for item in raw_data['list']:
            forecast = {
                'date': datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d %H:%M'),
                'temperature': item['main']['temp'],
                'weather': item['weather'][0]['description'],
                'humidity': item['main']['humidity']
            }
            forecasts.append(forecast)
        
        return {
            'city': raw_data['city']['name'],
            'country': raw_data['city']['country'],
            'forecasts': forecasts
        }

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    current = fetcher.get_current_weather("London", "UK")
    if current:
        print("Current Weather in London:")
        print(json.dumps(current, indent=2))
    
    forecast = fetcher.get_forecast("London", 3)
    if forecast:
        print("\n3-Day Forecast:")
        print(json.dumps(forecast, indent=2))

if __name__ == "__main__":
    main()