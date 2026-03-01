import requests
import json
import sys

def get_weather(api_key, city):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(data):
    if data is None:
        print("No data to display.")
        return
    try:
        city = data['name']
        country = data['sys']['country']
        temp = data['main']['temp']
        feels_like = data['main']['feels_like']
        humidity = data['main']['humidity']
        weather_desc = data['weather'][0]['description']
        wind_speed = data['wind']['speed']
        
        print(f"Weather in {city}, {country}:")
        print(f"  Temperature: {temp}°C (feels like {feels_like}°C)")
        print(f"  Conditions: {weather_desc}")
        print(f"  Humidity: {humidity}%")
        print(f"  Wind Speed: {wind_speed} m/s")
    except KeyError as e:
        print(f"Unexpected data structure: missing key {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather.py <api_key> <city>")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])
    
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)
import requests
import json
import os
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        
    def get_weather(self, city_name, country_code=''):
        if not self.api_key:
            raise ValueError("API key not provided. Set OPENWEATHER_API_KEY environment variable.")
        
        query = f"{city_name},{country_code}" if country_code else city_name
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'weather': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
            }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def display_weather(self, weather_data):
        if not weather_data:
            print("No weather data available.")
            return
        
        print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
        print(f"  Temperature: {weather_data['temperature']}°C")
        print(f"  Feels like: {weather_data['feels_like']}°C")
        print(f"  Conditions: {weather_data['weather'].title()}")
        print(f"  Humidity: {weather_data['humidity']}%")
        print(f"  Pressure: {weather_data['pressure']} hPa")
        print(f"  Wind Speed: {weather_data['wind_speed']} m/s")
        print(f"  Last Updated: {weather_data['timestamp']}")

def main():
    fetcher = WeatherFetcher()
    
    cities = ['London', 'New York', 'Tokyo', 'Paris']
    
    for city in cities:
        print(f"\nFetching weather for {city}...")
        weather = fetcher.get_weather(city)
        if weather:
            fetcher.display_weather(weather)
        else:
            print(f"Failed to fetch weather for {city}")

if __name__ == "__main__":
    main()
import requests
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

class WeatherFetcher:
    CACHE_DIR = Path.home() / '.weather_cache'
    CACHE_DURATION = 3600  # 1 hour in seconds
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        self.CACHE_DIR.mkdir(exist_ok=True)
    
    def _get_cache_path(self, location):
        safe_name = location.lower().replace(' ', '_').replace(',', '')
        return self.CACHE_DIR / f"{safe_name}.json"
    
    def _is_cache_valid(self, cache_path):
        if not cache_path.exists():
            return False
        
        cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - cache_time < timedelta(seconds=self.CACHE_DURATION)
    
    def _load_from_cache(self, cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def _save_to_cache(self, cache_path, data):
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except IOError:
            pass
    
    def fetch_weather(self, location, units='metric'):
        if not self.api_key:
            raise ValueError("API key is required")
        
        cache_path = self._get_cache_path(location)
        
        if self._is_cache_valid(cache_path):
            cached_data = self._load_from_cache(cache_path)
            if cached_data:
                cached_data['source'] = 'cache'
                return cached_data
        
        params = {
            'q': location,
            'appid': self.api_key,
            'units': units
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            processed_data = {
                'location': data.get('name'),
                'country': data.get('sys', {}).get('country'),
                'temperature': data.get('main', {}).get('temp'),
                'feels_like': data.get('main', {}).get('feels_like'),
                'humidity': data.get('main', {}).get('humidity'),
                'pressure': data.get('main', {}).get('pressure'),
                'weather': data.get('weather', [{}])[0].get('description'),
                'wind_speed': data.get('wind', {}).get('speed'),
                'wind_deg': data.get('wind', {}).get('deg'),
                'clouds': data.get('clouds', {}).get('all'),
                'visibility': data.get('visibility'),
                'timestamp': data.get('dt'),
                'timezone': data.get('timezone'),
                'source': 'api'
            }
            
            self._save_to_cache(cache_path, processed_data)
            return processed_data
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch weather data: {e}")
        except (KeyError, IndexError) as e:
            raise ValueError(f"Invalid response format: {e}")
    
    def format_weather_output(self, weather_data):
        if not weather_data:
            return "No weather data available"
        
        output_lines = []
        output_lines.append(f"Weather for {weather_data['location']}, {weather_data['country']}")
        output_lines.append(f"Temperature: {weather_data['temperature']}°C (feels like {weather_data['feels_like']}°C)")
        output_lines.append(f"Conditions: {weather_data['weather'].title()}")
        output_lines.append(f"Humidity: {weather_data['humidity']}%")
        output_lines.append(f"Pressure: {weather_data['pressure']} hPa")
        output_lines.append(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_deg']}°")
        output_lines.append(f"Cloud cover: {weather_data['clouds']}%")
        output_lines.append(f"Visibility: {weather_data['visibility']} meters")
        output_lines.append(f"Source: {weather_data['source']}")
        
        return '\n'.join(output_lines)

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python fetch_weather.py <location> [units]")
        print("Example: python fetch_weather.py 'London,UK' metric")
        sys.exit(1)
    
    location = sys.argv[1]
    units = sys.argv[2] if len(sys.argv) > 2 else 'metric'
    
    api_key = "YOUR_API_KEY_HERE"
    
    if api_key == "YOUR_API_KEY_HERE":
        print("Error: Please set your OpenWeatherMap API key in the script")
        sys.exit(1)
    
    fetcher = WeatherFetcher(api_key)
    
    try:
        weather_data = fetcher.fetch_weather(location, units)
        print(fetcher.format_weather_output(weather_data))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()