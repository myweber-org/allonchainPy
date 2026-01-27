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

def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather_data.py <API_KEY> <CITY_NAME>")
        sys.exit(1)

    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])

    weather_data = get_weather(api_key, city)
    display_weather(weather_data)

if __name__ == "__main__":
    main()
import requests
import json
import time
from datetime import datetime, timedelta
import os
from typing import Optional, Dict, Any

CACHE_FILE = "weather_cache.json"
CACHE_DURATION = 300  # 5 minutes in seconds

def load_cache() -> Dict[str, Any]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except (json.JSONDecodeError, IOError):
            pass
    return {}

def save_cache(cache: Dict[str, Any]) -> None:
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except IOError:
        pass

def is_cache_valid(timestamp: float) -> bool:
    current_time = time.time()
    return (current_time - timestamp) < CACHE_DURATION

def fetch_weather_data(city: str, api_key: str) -> Optional[Dict[str, Any]]:
    cache = load_cache()
    
    if city in cache:
        cached_data = cache[city]
        if is_cache_valid(cached_data.get('timestamp', 0)):
            print(f"Using cached data for {city}")
            return cached_data.get('data')
    
    url = f"http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('cod') != 200:
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None
        
        result = {
            'city': data['name'],
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'timestamp': time.time()
        }
        
        cache[city] = {
            'timestamp': time.time(),
            'data': result
        }
        save_cache(cache)
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"Data parsing error: {e}")
        return None

def display_weather(data: Dict[str, Any]) -> None:
    if not data:
        print("No weather data available")
        return
    
    print(f"Weather in {data['city']}:")
    print(f"  Temperature: {data['temperature']}°C")
    print(f"  Humidity: {data['humidity']}%")
    print(f"  Conditions: {data['description']}")
    print(f"  Wind Speed: {data['wind_speed']} m/s")
    
    cache_time = datetime.fromtimestamp(data['timestamp'])
    print(f"  Data fetched: {cache_time.strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    api_key = os.environ.get('WEATHER_API_KEY')
    if not api_key:
        print("Please set WEATHER_API_KEY environment variable")
        return
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        print(f"\nFetching weather for {city}...")
        weather_data = fetch_weather_data(city, api_key)
        if weather_data:
            display_weather(weather_data)
        else:
            print(f"Failed to fetch weather data for {city}")
        time.sleep(1)

if __name__ == "__main__":
    main()