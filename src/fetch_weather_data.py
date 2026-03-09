import requests
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import hashlib
import os

class WeatherFetcher:
    """Fetches weather data from OpenWeatherMap API with caching."""
    
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    CACHE_DIR = ".weather_cache"
    CACHE_DURATION = 300  # 5 minutes in seconds
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        if not os.path.exists(self.CACHE_DIR):
            os.makedirs(self.CACHE_DIR)
    
    def _get_cache_key(self, city: str, country_code: str) -> str:
        key_string = f"{city.lower()}_{country_code.lower()}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        return os.path.join(self.CACHE_DIR, f"{cache_key}.json")
    
    def _is_cache_valid(self, cache_path: str) -> bool:
        if not os.path.exists(cache_path):
            return False
        
        file_mtime = os.path.getmtime(cache_path)
        current_time = time.time()
        return (current_time - file_mtime) < self.CACHE_DURATION
    
    def _read_from_cache(self, cache_path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    
    def _write_to_cache(self, cache_path: str, data: Dict[str, Any]):
        with open(cache_path, 'w') as f:
            json.dump(data, f)
    
    def fetch_weather(self, city: str, country_code: str = "us") -> Dict[str, Any]:
        cache_key = self._get_cache_key(city, country_code)
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_cache_valid(cache_path):
            cached_data = self._read_from_cache(cache_path)
            if cached_data:
                cached_data['source'] = 'cache'
                return cached_data
        
        params = {
            'q': f"{city},{country_code}",
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            processed_data = {
                'city': data.get('name'),
                'country': data.get('sys', {}).get('country'),
                'temperature': data.get('main', {}).get('temp'),
                'feels_like': data.get('main', {}).get('feels_like'),
                'humidity': data.get('main', {}).get('humidity'),
                'pressure': data.get('main', {}).get('pressure'),
                'weather': data.get('weather', [{}])[0].get('description'),
                'wind_speed': data.get('wind', {}).get('speed'),
                'timestamp': datetime.now().isoformat(),
                'source': 'api'
            }
            
            self._write_to_cache(cache_path, processed_data)
            return processed_data
            
        except requests.exceptions.RequestException as e:
            return {
                'error': str(e),
                'city': city,
                'country': country_code,
                'timestamp': datetime.now().isoformat(),
                'source': 'error'
            }
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return {
                'error': f"Data parsing error: {str(e)}",
                'city': city,
                'country': country_code,
                'timestamp': datetime.now().isoformat(),
                'source': 'error'
            }
    
    def get_weather_summary(self, city: str, country_code: str = "us") -> str:
        data = self.fetch_weather(city, country_code)
        
        if 'error' in data:
            return f"Error fetching weather for {city}, {country_code}: {data['error']}"
        
        source_note = " (cached)" if data.get('source') == 'cache' else ""
        
        summary = f"""
Weather in {data['city']}, {data['country']}{source_note}:
Temperature: {data['temperature']}°C (feels like {data['feels_like']}°C)
Conditions: {data['weather']}
Humidity: {data['humidity']}%
Pressure: {data['pressure']} hPa
Wind Speed: {data['wind_speed']} m/s
Last Updated: {data['timestamp']}
"""
        return summary.strip()

def main():
    api_key = os.environ.get("OPENWEATHER_API_KEY", "your_api_key_here")
    
    if api_key == "your_api_key_here":
        print("Please set OPENWEATHER_API_KEY environment variable")
        return
    
    fetcher = WeatherFetcher(api_key)
    
    cities = [
        ("London", "uk"),
        ("New York", "us"),
        ("Tokyo", "jp"),
        ("Sydney", "au")
    ]
    
    for city, country in cities:
        print(fetcher.get_weather_summary(city, country))
        print("-" * 50)
        
        time.sleep(1)

if __name__ == "__main__":
    main()