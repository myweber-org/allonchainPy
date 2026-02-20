
import requests
import json
from datetime import datetime

def get_weather_data(api_key, city_name):
    """
    Fetch current weather data for a given city using OpenWeatherMap API.
    
    Args:
        api_key (str): OpenWeatherMap API key
        city_name (str): Name of the city to get weather for
    
    Returns:
        dict: Dictionary containing weather data or error information
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('cod') != 200:
            return {
                'success': False,
                'error': data.get('message', 'Unknown error')
            }
        
        weather_info = {
            'success': True,
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['main'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'wind_deg': data['wind']['deg'],
            'visibility': data.get('visibility', 'N/A'),
            'clouds': data['clouds']['all'],
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return weather_info
        
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': f"Network error: {str(e)}"
        }
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return {
            'success': False,
            'error': f"Data parsing error: {str(e)}"
        }

def display_weather(weather_data):
    """
    Display weather data in a formatted way.
    
    Args:
        weather_data (dict): Weather data dictionary from get_weather_data
    """
    if not weather_data.get('success'):
        print(f"Error: {weather_data.get('error', 'Unknown error')}")
        return
    
    print("\n" + "="*50)
    print(f"Weather Report for {weather_data['city']}, {weather_data['country']}")
    print("="*50)
    print(f"Current Time: {weather_data['timestamp']}")
    print(f"Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)")
    print(f"Weather: {weather_data['weather']} - {weather_data['description']}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_deg']}°")
    print(f"Cloudiness: {weather_data['clouds']}%")
    print(f"Visibility: {weather_data['visibility']} meters")
    print(f"Sunrise: {weather_data['sunrise']}")
    print(f"Sunset: {weather_data['sunset']}")
    print("="*50)

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    CITY = "London"
    
    print(f"Fetching weather data for {CITY}...")
    weather = get_weather_data(API_KEY, CITY)
    display_weather(weather)import requests

def get_weather(city_name, api_key):
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
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(weather_data):
    if weather_data:
        city = weather_data['name']
        temp = weather_data['main']['temp']
        description = weather_data['weather'][0]['description']
        humidity = weather_data['main']['humidity']
        print(f"Weather in {city}:")
        print(f"  Temperature: {temp}°C")
        print(f"  Conditions: {description}")
        print(f"  Humidity: {humidity}%")
    else:
        print("No weather data to display.")

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    city = input("Enter city name: ")
    weather = get_weather(city, API_KEY)
    display_weather(weather)import requests
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

class WeatherFetcher:
    """A class to fetch and cache weather data from OpenWeatherMap API."""
    
    def __init__(self, api_key: str, cache_duration: int = 300):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        self.cache_duration = cache_duration
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    def get_weather(self, city: str, country_code: Optional[str] = None) -> Dict[str, Any]:
        """Fetch weather data for a given city with optional country code."""
        location_key = f"{city.lower()}_{country_code.lower()}" if country_code else city.lower()
        
        # Check cache first
        cached_data = self._get_from_cache(location_key)
        if cached_data:
            return cached_data
        
        # Build query parameters
        params = {
            "q": f"{city},{country_code}" if country_code else city,
            "appid": self.api_key,
            "units": "metric"
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract relevant information
            processed_data = {
                "location": data.get("name", city),
                "country": data.get("sys", {}).get("country", ""),
                "temperature": data.get("main", {}).get("temp"),
                "feels_like": data.get("main", {}).get("feels_like"),
                "humidity": data.get("main", {}).get("humidity"),
                "pressure": data.get("main", {}).get("pressure"),
                "description": data.get("weather", [{}])[0].get("description", ""),
                "wind_speed": data.get("wind", {}).get("speed"),
                "wind_direction": data.get("wind", {}).get("deg"),
                "cloudiness": data.get("clouds", {}).get("all"),
                "visibility": data.get("visibility"),
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the result
            self._add_to_cache(location_key, processed_data)
            return processed_data
            
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Network error: {str(e)}",
                "location": city,
                "timestamp": datetime.now().isoformat()
            }
        except json.JSONDecodeError:
            return {
                "error": "Invalid response from API",
                "location": city,
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from cache if it exists and is still valid."""
        if key in self.cache:
            cached_entry = self.cache[key]
            cache_time = datetime.fromisoformat(cached_entry["cache_timestamp"])
            if datetime.now() - cache_time < timedelta(seconds=self.cache_duration):
                return cached_entry["data"]
        return None
    
    def _add_to_cache(self, key: str, data: Dict[str, Any]) -> None:
        """Add data to cache with timestamp."""
        self.cache[key] = {
            "data": data,
            "cache_timestamp": datetime.now().isoformat()
        }
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.cache.clear()

def display_weather(weather_data: Dict[str, Any]) -> None:
    """Display weather data in a readable format."""
    if "error" in weather_data:
        print(f"Error for {weather_data.get('location', 'Unknown')}: {weather_data['error']}")
        return
    
    print("\n" + "="*50)
    print(f"Weather for {weather_data['location']}, {weather_data['country']}")
    print("="*50)
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels like: {weather_data['feels_like']}°C")
    print(f"Conditions: {weather_data['description'].title()}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
    print(f"Cloudiness: {weather_data['cloudiness']}%")
    if weather_data.get('visibility'):
        print(f"Visibility: {weather_data['visibility'] / 1000:.1f} km")
    print(f"Last updated: {weather_data['timestamp']}")
    print("="*50)

# Example usage
if __name__ == "__main__":
    # Replace with your actual API key from OpenWeatherMap
    API_KEY = "your_api_key_here"
    
    fetcher = WeatherFetcher(API_KEY)
    
    # Fetch weather for multiple cities
    locations = [
        ("London", "UK"),
        ("New York", "US"),
        ("Tokyo", "JP"),
        ("Paris", "FR")
    ]
    
    for city, country in locations:
        weather = fetcher.get_weather(city, country)
        display_weather(weather)
        time.sleep(1)  # Be nice to the APIimport requests
import json
import os

def get_weather(city_name, api_key=None):
    """
    Fetch current weather data for a given city.
    
    Args:
        city_name (str): Name of the city
        api_key (str): OpenWeatherMap API key. If None, reads from WEATHER_API_KEY env variable.
    
    Returns:
        dict: Weather data if successful, None otherwise
    """
    if api_key is None:
        api_key = os.getenv('WEATHER_API_KEY')
    
    if not api_key:
        print("API key not provided. Set WEATHER_API_KEY environment variable.")
        return None
    
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('cod') != 200:
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None
        
        return {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'wind_direction': data['wind'].get('deg', 0),
            'visibility': data.get('visibility', 0),
            'clouds': data['clouds']['all']
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Data parsing error: {e}")
        return None

def display_weather(weather_data):
    """
    Display weather data in a readable format.
    
    Args:
        weather_data (dict): Weather data dictionary
    """
    if not weather_data:
        print("No weather data to display.")
        return
    
    print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
    print(f"  Temperature: {weather_data['temperature']}°C")
    print(f"  Feels like: {weather_data['feels_like']}°C")
    print(f"  Weather: {weather_data['weather'].title()}")
    print(f"  Humidity: {weather_data['humidity']}%")
    print(f"  Pressure: {weather_data['pressure']} hPa")
    print(f"  Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
    print(f"  Visibility: {weather_data['visibility']} meters")
    print(f"  Cloud cover: {weather_data['clouds']}%")

if __name__ == "__main__":
    # Example usage
    city = "London"
    weather = get_weather(city)
    
    if weather:
        display_weather(weather)
    else:
        print(f"Failed to fetch weather data for {city}")