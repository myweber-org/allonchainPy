
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import hashlib
import os

class WeatherFetcher:
    def __init__(self, api_key: str, cache_dir: str = "./weather_cache"):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.cache_dir = cache_dir
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _get_cache_key(self, city: str) -> str:
        return hashlib.md5(city.lower().encode()).hexdigest()
    
    def _read_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                cache_time = datetime.fromisoformat(cached_data['cached_at'])
                if datetime.now() - cache_time < timedelta(minutes=30):
                    return cached_data['data']
        return None
    
    def _write_cache(self, cache_key: str, data: Dict[str, Any]):
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        cache_data = {
            'cached_at': datetime.now().isoformat(),
            'data': data
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    
    def fetch_weather(self, city: str) -> Dict[str, Any]:
        cache_key = self._get_cache_key(city)
        cached_data = self._read_cache(cache_key)
        
        if cached_data:
            print(f"Using cached data for {city}")
            return cached_data
        
        print(f"Fetching fresh data for {city}")
        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
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
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
            }
            
            self._write_cache(cache_key, processed_data)
            return processed_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return {'error': str(e)}
        except KeyError as e:
            print(f"Unexpected API response format: {e}")
            return {'error': 'Invalid API response'}
    
    def display_weather(self, weather_data: Dict[str, Any]):
        if 'error' in weather_data:
            print(f"Error: {weather_data['error']}")
            return
        
        print("\n" + "="*40)
        print(f"Weather in {weather_data['city']}, {weather_data['country']}")
        print("="*40)
        print(f"Temperature: {weather_data['temperature']}°C")
        print(f"Feels like: {weather_data['feels_like']}°C")
        print(f"Weather: {weather_data['weather']} - {weather_data['description']}")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Pressure: {weather_data['pressure']} hPa")
        print(f"Wind Speed: {weather_data['wind_speed']} m/s")
        print(f"Last Updated: {weather_data['timestamp']}")
        print("="*40)

def main():
    api_key = os.environ.get('OPENWEATHER_API_KEY')
    if not api_key:
        print("Please set OPENWEATHER_API_KEY environment variable")
        return
    
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris", "Sydney"]
    
    for city in cities:
        weather_data = fetcher.fetch_weather(city)
        fetcher.display_weather(weather_data)
        time.sleep(1)

if __name__ == "__main__":
    main()