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
        """Create cache directory if it doesn't exist."""
        if not os.path.exists(self.CACHE_DIR):
            os.makedirs(self.CACHE_DIR)
    
    def _get_cache_key(self, city: str, country_code: str) -> str:
        """Generate cache key from city and country."""
        key_string = f"{city.lower()}_{country_code.lower()}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get full path to cache file."""
        return os.path.join(self.CACHE_DIR, f"{cache_key}.json")
    
    def _is_cache_valid(self, cache_path: str) -> bool:
        """Check if cache file exists and is still valid."""
        if not os.path.exists(cache_path):
            return False
        
        file_mtime = os.path.getmtime(cache_path)
        current_time = time.time()
        
        return (current_time - file_mtime) < self.CACHE_DURATION
    
    def _read_from_cache(self, cache_path: str) -> Optional[Dict[str, Any]]:
        """Read data from cache file."""
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def _write_to_cache(self, cache_path: str, data: Dict[str, Any]):
        """Write data to cache file."""
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except IOError:
            pass  # Silently fail if cache write fails
    
    def fetch_weather(self, city: str, country_code: str = "us") -> Dict[str, Any]:
        """Fetch weather data for a city."""
        cache_key = self._get_cache_key(city, country_code)
        cache_path = self._get_cache_path(cache_key)
        
        # Try cache first
        if self._is_cache_valid(cache_path):
            cached_data = self._read_from_cache(cache_path)
            if cached_data:
                cached_data['source'] = 'cache'
                return cached_data
        
        # Fetch from API
        params = {
            'q': f"{city},{country_code}",
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Process and enrich data
            processed_data = self._process_weather_data(data)
            processed_data['source'] = 'api'
            processed_data['timestamp'] = datetime.now().isoformat()
            
            # Cache the result
            self._write_to_cache(cache_path, processed_data)
            
            return processed_data
            
        except requests.exceptions.RequestException as e:
            return {
                'error': True,
                'message': f"Failed to fetch weather data: {str(e)}",
                'city': city,
                'country': country_code
            }
        except json.JSONDecodeError:
            return {
                'error': True,
                'message': "Invalid response from weather API",
                'city': city,
                'country': country_code
            }
    
    def _process_weather_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and enrich raw weather data."""
        processed = {
            'city': raw_data.get('name', 'Unknown'),
            'country': raw_data.get('sys', {}).get('country', 'Unknown'),
            'temperature': raw_data.get('main', {}).get('temp'),
            'feels_like': raw_data.get('main', {}).get('feels_like'),
            'humidity': raw_data.get('main', {}).get('humidity'),
            'pressure': raw_data.get('main', {}).get('pressure'),
            'weather': raw_data.get('weather', [{}])[0].get('description', 'Unknown'),
            'wind_speed': raw_data.get('wind', {}).get('speed'),
            'wind_direction': raw_data.get('wind', {}).get('deg'),
            'visibility': raw_data.get('visibility'),
            'cloudiness': raw_data.get('clouds', {}).get('all'),
            'sunrise': raw_data.get('sys', {}).get('sunrise'),
            'sunset': raw_data.get('sys', {}).get('sunset'),
            'timezone': raw_data.get('timezone'),
            'coordinates': raw_data.get('coord', {}),
            'error': False
        }
        
        # Add temperature in Fahrenheit
        if processed['temperature'] is not None:
            processed['temperature_f'] = (processed['temperature'] * 9/5) + 32
        
        # Add wind direction as compass point
        if processed['wind_direction'] is not None:
            processed['wind_direction_compass'] = self._degrees_to_compass(
                processed['wind_direction']
            )
        
        return processed
    
    @staticmethod
    def _degrees_to_compass(degrees: float) -> str:
        """Convert wind direction in degrees to compass direction."""
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                     'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        index = round(degrees / 22.5) % 16
        return directions[index]
    
    def clear_cache(self, older_than_hours: int = 24):
        """Clear cache files older than specified hours."""
        cutoff_time = time.time() - (older_than_hours * 3600)
        
        for filename in os.listdir(self.CACHE_DIR):
            filepath = os.path.join(self.CACHE_DIR, filename)
            if os.path.isfile(filepath):
                if os.path.getmtime(filepath) < cutoff_time:
                    os.remove(filepath)


def example_usage():
    """Example of how to use the WeatherFetcher class."""
    # Replace with your actual API key
    API_KEY = "your_api_key_here"
    
    fetcher = WeatherFetcher(API_KEY)
    
    # Fetch weather for some cities
    cities = [
        ("London", "uk"),
        ("New York", "us"),
        ("Tokyo", "jp"),
        ("Sydney", "au")
    ]
    
    for city, country in cities:
        print(f"\nFetching weather for {city}, {country}...")
        weather = fetcher.fetch_weather(city, country)
        
        if weather.get('error'):
            print(f"Error: {weather['message']}")
        else:
            print(f"Source: {weather['source']}")
            print(f"Weather: {weather['weather']}")
            print(f"Temperature: {weather['temperature']}°C ({weather.get('temperature_f', 'N/A')}°F)")
            print(f"Humidity: {weather['humidity']}%")
            print(f"Wind: {weather['wind_speed']} m/s {weather.get('wind_direction_compass', '')}")


if __name__ == "__main__":
    example_usage()