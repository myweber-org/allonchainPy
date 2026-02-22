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
        print("Usage: python fetch_weather_data.py <api_key> <city>")
        sys.exit(1)

    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)
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
        print(f"Error fetching weather data: {e}", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}", file=sys.stderr)
        return None

def display_weather(data):
    if data and data.get('cod') == 200:
        city = data['name']
        country = data['sys']['country']
        temp = data['main']['temp']
        weather_desc = data['weather'][0]['description']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']

        print(f"Weather in {city}, {country}:")
        print(f"  Temperature: {temp}°C")
        print(f"  Conditions: {weather_desc}")
        print(f"  Humidity: {humidity}%")
        print(f"  Wind Speed: {wind_speed} m/s")
    else:
        error_msg = data.get('message', 'Unknown error') if data else 'No data received'
        print(f"Could not retrieve weather data. Error: {error_msg}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather_data.py <api_key> <city>")
        sys.exit(1)

    api_key = sys.argv[1]
    city = sys.argv[2]
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)import requests
import json
import os

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

def display_weather(data):
    if data and data.get('cod') == 200:
        city = data['name']
        country = data['sys']['country']
        temp = data['main']['temp']
        weather_desc = data['weather'][0]['description']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']
        
        print(f"Weather in {city}, {country}:")
        print(f"  Temperature: {temp}°C")
        print(f"  Conditions: {weather_desc}")
        print(f"  Humidity: {humidity}%")
        print(f"  Wind Speed: {wind_speed} m/s")
    else:
        print("Could not retrieve weather data.")

def main():
    api_key = os.environ.get('OPENWEATHER_API_KEY')
    if not api_key:
        print("Please set the OPENWEATHER_API_KEY environment variable.")
        return
    
    city = input("Enter city name: ")
    weather_data = get_weather(city, api_key)
    display_weather(weather_data)

if __name__ == "__main__":
    main()import requests
import json

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
        weather_desc = data['weather'][0]['description']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']
        
        print(f"Weather in {city}, {country}:")
        print(f"  Temperature: {temp}°C")
        print(f"  Conditions: {weather_desc}")
        print(f"  Humidity: {humidity}%")
        print(f"  Wind Speed: {wind_speed} m/s")
    else:
        print("City not found or invalid data received.")

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    city_name = input("Enter city name: ")
    weather_data = get_weather(API_KEY, city_name)
    display_weather(weather_data)import requests
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'WeatherApp/1.0'})

    def get_current_weather(self, city_name, country_code=None):
        """Fetch current weather data for a given city."""
        try:
            query = city_name
            if country_code:
                query += f",{country_code}"
            
            params = {
                'q': query,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = self.session.get(
                f"{self.base_url}/weather",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching weather: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            return None
        except KeyError as e:
            logger.error(f"Missing expected data in response: {e}")
            return None

    def _parse_weather_data(self, raw_data):
        """Parse and structure raw weather data."""
        return {
            'city': raw_data.get('name'),
            'country': raw_data.get('sys', {}).get('country'),
            'temperature': raw_data.get('main', {}).get('temp'),
            'feels_like': raw_data.get('main', {}).get('feels_like'),
            'humidity': raw_data.get('main', {}).get('humidity'),
            'pressure': raw_data.get('main', {}).get('pressure'),
            'weather_description': raw_data.get('weather', [{}])[0].get('description'),
            'wind_speed': raw_data.get('wind', {}).get('speed'),
            'wind_direction': raw_data.get('wind', {}).get('deg'),
            'visibility': raw_data.get('visibility'),
            'cloudiness': raw_data.get('clouds', {}).get('all'),
            'sunrise': self._timestamp_to_datetime(raw_data.get('sys', {}).get('sunrise')),
            'sunset': self._timestamp_to_datetime(raw_data.get('sys', {}).get('sunset')),
            'data_timestamp': self._timestamp_to_datetime(raw_data.get('dt')),
            'raw_data': raw_data
        }

    def _timestamp_to_datetime(self, timestamp):
        """Convert Unix timestamp to datetime string."""
        if timestamp:
            return datetime.utcfromtimestamp(timestamp).isoformat()
        return None

    def get_weather_forecast(self, city_name, days=5):
        """Fetch weather forecast for multiple days."""
        try:
            params = {
                'q': city_name,
                'appid': self.api_key,
                'units': 'metric',
                'cnt': days * 8  # 8 forecasts per day (3-hour intervals)
            }
            
            response = self.session.get(
                f"{self.base_url}/forecast",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            return self._parse_forecast_data(data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching forecast: {e}")
            return None

    def _parse_forecast_data(self, raw_data):
        """Parse and structure forecast data."""
        forecasts = []
        city_info = raw_data.get('city', {})
        
        for item in raw_data.get('list', []):
            forecast = {
                'timestamp': self._timestamp_to_datetime(item.get('dt')),
                'temperature': item.get('main', {}).get('temp'),
                'feels_like': item.get('main', {}).get('feels_like'),
                'humidity': item.get('main', {}).get('humidity'),
                'weather_description': item.get('weather', [{}])[0].get('description'),
                'wind_speed': item.get('wind', {}).get('speed'),
                'precipitation_probability': item.get('pop', 0),
                'rain_volume': item.get('rain', {}).get('3h', 0),
                'snow_volume': item.get('snow', {}).get('3h', 0)
            }
            forecasts.append(forecast)
        
        return {
            'city': city_info.get('name'),
            'country': city_info.get('country'),
            'forecasts': forecasts
        }

def display_weather_summary(weather_data):
    """Display a formatted summary of weather data."""
    if not weather_data:
        print("No weather data available.")
        return
    
    print("\n" + "="*50)
    print(f"Weather in {weather_data['city']}, {weather_data['country']}")
    print("="*50)
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels like: {weather_data['feels_like']}°C")
    print(f"Conditions: {weather_data['weather_description'].title()}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Wind: {weather_data['wind_speed']} m/s")
    print(f"Visibility: {weather_data['visibility']} meters")
    print(f"Sunrise: {weather_data['sunrise']}")
    print(f"Sunset: {weather_data['sunset']}")
    print("="*50)

def main():
    """Example usage of the WeatherFetcher class."""
    # Note: In production, use environment variables for API keys
    API_KEY = "your_api_key_here"  # Replace with actual API key
    
    fetcher = WeatherFetcher(API_KEY)
    
    # Fetch current weather
    weather = fetcher.get_current_weather("London", "UK")
    if weather:
        display_weather_summary(weather)
        
        # Save to JSON file
        with open('weather_data.json', 'w') as f:
            json.dump(weather, f, indent=2)
        logger.info("Weather data saved to weather_data.json")
    
    # Fetch forecast
    forecast = fetcher.get_weather_forecast("London", days=3)
    if forecast:
        print(f"\n3-day forecast for {forecast['city']}:")
        for i, day_forecast in enumerate(forecast['forecasts'][:8:3], 1):
            print(f"Day {i}: {day_forecast['temperature']}°C - {day_forecast['weather_description']}")

if __name__ == "__main__":
    main()