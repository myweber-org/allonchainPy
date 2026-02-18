
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

class WeatherFetcher:
    def __init__(self, api_key: str, cache_duration: int = 300):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        self.cache_duration = cache_duration
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    def get_weather(self, city: str) -> Optional[Dict[str, Any]]:
        cache_key = city.lower()
        
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_duration:
                return cached_data['data']
        
        try:
            params = {
                'q': city,
                'appid': self.api_key,
                'units': 'metric'
            }
            
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
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'wind_direction': data['wind'].get('deg', 0),
                'visibility': data.get('visibility', 0),
                'clouds': data['clouds']['all'],
                'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).isoformat(),
                'sunset': datetime.fromtimestamp(data['sys']['sunset']).isoformat(),
                'timestamp': datetime.now().isoformat()
            }
            
            self.cache[cache_key] = {
                'data': processed_data,
                'timestamp': time.time()
            }
            
            return processed_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"Error processing weather data: {e}")
            return None
    
    def clear_cache(self) -> None:
        self.cache.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        return {
            'cache_size': len(self.cache),
            'cached_cities': list(self.cache.keys()),
            'cache_duration': self.cache_duration
        }

def display_weather(weather_data: Dict[str, Any]) -> None:
    if not weather_data:
        print("No weather data available")
        return
    
    print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels like: {weather_data['feels_like']}°C")
    print(f"Conditions: {weather_data['description'].title()}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
    print(f"Visibility: {weather_data['visibility']} meters")
    print(f"Cloud cover: {weather_data['clouds']}%")
    print(f"Sunrise: {weather_data['sunrise']}")
    print(f"Sunset: {weather_data['sunset']}")
    print(f"Data fetched at: {weather_data['timestamp']}")

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    
    fetcher = WeatherFetcher(API_KEY)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        print("\n" + "="*50)
        weather = fetcher.get_weather(city)
        display_weather(weather)
        time.sleep(1)
    
    print("\nCache Information:")
    print(json.dumps(fetcher.get_cache_info(), indent=2))