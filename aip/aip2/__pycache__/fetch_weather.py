
import requests
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

class WeatherFetcher:
    def __init__(self, api_key, cache_dir="weather_cache"):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        
    def _get_cache_path(self, city):
        return self.cache_dir / f"{city.lower().replace(' ', '_')}.json"
    
    def _is_cache_valid(self, cache_path):
        if not cache_path.exists():
            return False
        cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - cache_time < timedelta(minutes=30)
    
    def fetch_weather(self, city):
        cache_path = self._get_cache_path(city)
        
        if self._is_cache_valid(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            if cache_path.exists():
                with open(cache_path, 'r') as f:
                    return json.load(f)
            raise
    
    def format_weather(self, weather_data):
        if weather_data.get('cod') != 200:
            return f"Error: {weather_data.get('message', 'Unknown error')}"
        
        main = weather_data['main']
        weather = weather_data['weather'][0]
        
        return {
            'city': weather_data['name'],
            'temperature': main['temp'],
            'feels_like': main['feels_like'],
            'humidity': main['humidity'],
            'pressure': main['pressure'],
            'description': weather['description'],
            'wind_speed': weather_data['wind']['speed'],
            'timestamp': datetime.fromtimestamp(weather_data['dt'])
        }

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        try:
            print(f"\nFetching weather for {city}...")
            weather_data = fetcher.fetch_weather(city)
            formatted = fetcher.format_weather(weather_data)
            
            if isinstance(formatted, dict):
                print(f"City: {formatted['city']}")
                print(f"Temperature: {formatted['temperature']}°C")
                print(f"Feels like: {formatted['feels_like']}°C")
                print(f"Description: {formatted['description']}")
                print(f"Humidity: {formatted['humidity']}%")
                print(f"Wind Speed: {formatted['wind_speed']} m/s")
                print(f"Last updated: {formatted['timestamp']}")
            else:
                print(formatted)
                
            time.sleep(1)
            
        except Exception as e:
            print(f"Failed to get weather for {city}: {e}")

if __name__ == "__main__":
    main()
import requests
import json
from datetime import datetime
import sys

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5/weather"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_weather(self, city_name, units="metric"):
        params = {
            "q": city_name,
            "appid": self.api_key,
            "units": units
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return self._parse_response(response.json())
        except requests.exceptions.RequestException as e:
            return {"error": f"Network error: {str(e)}"}
        except json.JSONDecodeError:
            return {"error": "Invalid response from server"}
        except KeyError as e:
            return {"error": f"Missing data in response: {str(e)}"}
    
    def _parse_response(self, data):
        return {
            "city": data.get("name", "Unknown"),
            "country": data.get("sys", {}).get("country", "Unknown"),
            "temperature": data.get("main", {}).get("temp"),
            "feels_like": data.get("main", {}).get("feels_like"),
            "humidity": data.get("main", {}).get("humidity"),
            "pressure": data.get("main", {}).get("pressure"),
            "weather": data.get("weather", [{}])[0].get("description", "Unknown"),
            "wind_speed": data.get("wind", {}).get("speed"),
            "wind_direction": data.get("wind", {}).get("deg"),
            "visibility": data.get("visibility"),
            "cloudiness": data.get("clouds", {}).get("all"),
            "sunrise": self._format_timestamp(data.get("sys", {}).get("sunrise")),
            "sunset": self._format_timestamp(data.get("sys", {}).get("sunset")),
            "timestamp": datetime.now().isoformat()
        }
    
    def _format_timestamp(self, timestamp):
        if timestamp:
            return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
        return "N/A"

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_weather.py <city_name>")
        sys.exit(1)
    
    city_name = " ".join(sys.argv[1:])
    api_key = "your_api_key_here"
    
    fetcher = WeatherFetcher(api_key)
    weather_data = fetcher.get_weather(city_name)
    
    if "error" in weather_data:
        print(f"Error: {weather_data['error']}")
        sys.exit(1)
    
    print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels like: {weather_data['feels_like']}°C")
    print(f"Weather: {weather_data['weather']}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Wind: {weather_data['wind_speed']} m/s")
    print(f"Sunrise: {weather_data['sunrise']}")
    print(f"Sunset: {weather_data['sunset']}")

if __name__ == "__main__":
    main()