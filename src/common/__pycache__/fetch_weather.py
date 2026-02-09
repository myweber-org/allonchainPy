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
    if data.get('cod') != 200:
        print(f"Error: {data.get('message', 'Unknown error')}")
        return

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

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather.py <API_KEY> <CITY_NAME>")
        sys.exit(1)

    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)import requests
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
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather.py <API_KEY> <CITY>")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = sys.argv[2]
    
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)import requests
import json
import time
from datetime import datetime, timedelta
import os

class WeatherFetcher:
    def __init__(self, api_key, cache_dir='./weather_cache'):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.cache_dir = cache_dir
        self.cache_duration = 300  # Cache for 5 minutes
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _get_cache_filename(self, city):
        return os.path.join(self.cache_dir, f"{city.lower().replace(' ', '_')}.json")
    
    def _is_cache_valid(self, cache_file):
        if not os.path.exists(cache_file):
            return False
        
        file_mtime = os.path.getmtime(cache_file)
        current_time = time.time()
        return (current_time - file_mtime) < self.cache_duration
    
    def _read_from_cache(self, cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def _write_to_cache(self, cache_file, data):
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except IOError:
            pass
    
    def fetch_weather(self, city):
        cache_file = self._get_cache_filename(city)
        
        if self._is_cache_valid(cache_file):
            cached_data = self._read_from_cache(cache_file)
            if cached_data:
                cached_data['source'] = 'cache'
                return cached_data
        
        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('cod') != 200:
                return {'error': f"API error: {data.get('message', 'Unknown error')}"}
            
            processed_data = {
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'weather': data['weather'][0]['main'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'timestamp': datetime.now().isoformat(),
                'source': 'api'
            }
            
            self._write_to_cache(cache_file, processed_data)
            return processed_data
            
        except requests.exceptions.RequestException as e:
            return {'error': f"Network error: {str(e)}"}
        except (KeyError, ValueError) as e:
            return {'error': f"Data parsing error: {str(e)}"}
    
    def get_weather_summary(self, city):
        weather_data = self.fetch_weather(city)
        
        if 'error' in weather_data:
            return f"Error fetching weather for {city}: {weather_data['error']}"
        
        source_note = " (cached)" if weather_data['source'] == 'cache' else ""
        
        summary = f"""
Weather in {weather_data['city']}, {weather_data['country']}{source_note}:
Temperature: {weather_data['temperature']}°C (feels like {weather_data['feels_like']}°C)
Conditions: {weather_data['weather']} - {weather_data['description']}
Humidity: {weather_data['humidity']}%
Pressure: {weather_data['pressure']} hPa
Wind Speed: {weather_data['wind_speed']} m/s
Last updated: {weather_data['timestamp']}
"""
        return summary.strip()

def main():
    api_key = os.environ.get('OPENWEATHER_API_KEY')
    
    if not api_key:
        print("Please set OPENWEATHER_API_KEY environment variable")
        return
    
    fetcher = WeatherFetcher(api_key)
    
    cities = ['London', 'New York', 'Tokyo', 'Paris']
    
    for city in cities:
        print(fetcher.get_weather_summary(city))
        print("-" * 50)

if __name__ == "__main__":
    main()