import requests
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WeatherFetcher:
    """Fetches weather data from OpenWeatherMap API"""
    
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    
    def __init__(self, api_key: str):
        """Initialize with API key"""
        self.api_key = api_key
        self.session = requests.Session()
        
    def get_weather(self, city: str, country_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetch current weather for a city"""
        try:
            query = f"{city},{country_code}" if country_code else city
            params = {
                'q': query,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            logger.info(f"Fetching weather for: {query}")
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {city}: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response for {city}: {str(e)}")
            return None
        except KeyError as e:
            logger.error(f"Unexpected data structure for {city}: {str(e)}")
            return None
    
    def _parse_weather_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse raw API response into structured format"""
        return {
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
            'sunrise': self._format_timestamp(raw_data.get('sys', {}).get('sunrise')),
            'sunset': self._format_timestamp(raw_data.get('sys', {}).get('sunset')),
            'timestamp': datetime.now().isoformat()
        }
    
    def _format_timestamp(self, timestamp: Optional[int]) -> Optional[str]:
        """Convert Unix timestamp to readable format"""
        if timestamp:
            return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        return None
    
    def display_weather(self, weather_data: Dict[str, Any]) -> None:
        """Display weather information in a readable format"""
        if not weather_data:
            print("No weather data available")
            return
            
        print("\n" + "="*50)
        print(f"Weather Report for {weather_data['city']}, {weather_data['country']}")
        print("="*50)
        print(f"Current Conditions: {weather_data['weather'].title()}")
        print(f"Temperature: {weather_data['temperature']}°C")
        print(f"Feels Like: {weather_data['feels_like']}°C")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Pressure: {weather_data['pressure']} hPa")
        print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
        print(f"Visibility: {weather_data['visibility']} meters")
        print(f"Cloudiness: {weather_data['cloudiness']}%")
        print(f"Sunrise: {weather_data['sunrise']}")
        print(f"Sunset: {weather_data['sunset']}")
        print(f"Report Time: {weather_data['timestamp']}")
        print("="*50)

def main():
    """Main execution function"""
    # Replace with your actual API key from https://openweathermap.org/api
    API_KEY = "your_api_key_here"
    
    if API_KEY == "your_api_key_here":
        logger.warning("Please replace 'your_api_key_here' with a valid OpenWeatherMap API key")
        return
    
    fetcher = WeatherFetcher(API_KEY)
    
    # Example cities to fetch weather for
    locations = [
        {"city": "London", "country": "GB"},
        {"city": "New York", "country": "US"},
        {"city": "Tokyo", "country": "JP"},
        {"city": "Sydney", "country": "AU"}
    ]
    
    for location in locations:
        weather = fetcher.get_weather(location["city"], location["country"])
        if weather:
            fetcher.display_weather(weather)
        else:
            print(f"\nFailed to fetch weather for {location['city']}")
        
        # Add a small delay between requests
        import time
        time.sleep(1)

if __name__ == "__main__":
    main()import requests
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
    if data and data.get('cod') == 200:
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
    else:
        error_msg = data.get('message', 'Unknown error') if data else 'No data received'
        print(f"Failed to retrieve weather data: {error_msg}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather.py <api_key> <city>")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = sys.argv[2]
    
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)
import requests
import json
import sys
from datetime import datetime

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
        
        if data['cod'] != 200:
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
            'timestamp': datetime.fromtimestamp(data['dt'])
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Data parsing error: {e}")
        return None

def display_weather(weather_data):
    if not weather_data:
        return
        
    print("\n" + "="*40)
    print(f"Weather in {weather_data['city']}, {weather_data['country']}")
    print("="*40)
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels like: {weather_data['feels_like']}°C")
    print(f"Weather: {weather_data['weather'].title()}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind Speed: {weather_data['wind_speed']} m/s")
    print(f"Last Updated: {weather_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*40)

def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather.py <api_key> <city>")
        print("Example: python fetch_weather.py abc123 London")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])
    
    print(f"Fetching weather for {city}...")
    weather_data = get_weather(api_key, city)
    
    if weather_data:
        display_weather(weather_data)
    else:
        print("Failed to fetch weather data")

if __name__ == "__main__":
    main()import requests
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
    if len(sys.argv) < 2:
        print("Usage: python fetch_weather.py <city_name>")
        print("Please set your API key in the OPENWEATHER_API_KEY environment variable.")
        sys.exit(1)

    city = sys.argv[1]
    api_key = "YOUR_API_KEY_HERE"

    import os
    env_key = os.environ.get('OPENWEATHER_API_KEY')
    if env_key:
        api_key = env_key

    if api_key == "YOUR_API_KEY_HERE":
        print("Error: API key not set.")
        print("Please set the OPENWEATHER_API_KEY environment variable or replace the placeholder in the script.")
        sys.exit(1)

    weather_data = get_weather(api_key, city)
    display_weather(weather_data)

if __name__ == "__main__":
    main()