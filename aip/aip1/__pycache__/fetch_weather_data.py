
import requests
import json
from datetime import datetime
import sys

class WeatherFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    def get_weather(self, city_name):
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return self._parse_weather_data(response.json())
        except requests.exceptions.RequestException as e:
            return f"Error fetching weather data: {str(e)}"
        except KeyError as e:
            return f"Unexpected API response format: {str(e)}"
    
    def _parse_weather_data(self, data):
        weather_info = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }
        return weather_info
    
    def display_weather(self, weather_data):
        if isinstance(weather_data, dict):
            print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
            print(f"  Temperature: {weather_data['temperature']}°C")
            print(f"  Feels like: {weather_data['feels_like']}°C")
            print(f"  Conditions: {weather_data['weather'].title()}")
            print(f"  Humidity: {weather_data['humidity']}%")
            print(f"  Pressure: {weather_data['pressure']} hPa")
            print(f"  Wind Speed: {weather_data['wind_speed']} m/s")
            print(f"  Last Updated: {weather_data['timestamp']}")
        else:
            print(weather_data)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_weather_data.py <city_name>")
        print("Example: python fetch_weather_data.py London")
        sys.exit(1)
    
    city_name = ' '.join(sys.argv[1:])
    
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    weather_data = fetcher.get_weather(city_name)
    fetcher.display_weather(weather_data)

if __name__ == "__main__":
    main()
import requests
import json
import os
from datetime import datetime

def get_weather_data(city_name, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def parse_weather_data(weather_json):
    if not weather_json or 'main' not in weather_json:
        return None
    
    weather_info = {
        'temperature': weather_json['main']['temp'],
        'feels_like': weather_json['main']['feels_like'],
        'humidity': weather_json['main']['humidity'],
        'pressure': weather_json['main']['pressure'],
        'description': weather_json['weather'][0]['description'],
        'wind_speed': weather_json['wind']['speed'],
        'city': weather_json['name'],
        'country': weather_json['sys']['country'],
        'timestamp': datetime.fromtimestamp(weather_json['dt'])
    }
    return weather_info

def save_weather_data(weather_info, filename='weather_data.json'):
    if not weather_info:
        return False
    
    try:
        with open(filename, 'a') as f:
            json.dump(weather_info, f, default=str)
            f.write('\n')
        return True
    except IOError as e:
        print(f"Error saving data: {e}")
        return False

def display_weather_info(weather_info):
    if not weather_info:
        print("No weather data available")
        return
    
    print(f"Weather in {weather_info['city']}, {weather_info['country']}:")
    print(f"Temperature: {weather_info['temperature']}°C")
    print(f"Feels like: {weather_info['feels_like']}°C")
    print(f"Conditions: {weather_info['description']}")
    print(f"Humidity: {weather_info['humidity']}%")
    print(f"Wind Speed: {weather_info['wind_speed']} m/s")
    print(f"Pressure: {weather_info['pressure']} hPa")
    print(f"Last updated: {weather_info['timestamp']}")

def main():
    api_key = os.environ.get('OPENWEATHER_API_KEY')
    if not api_key:
        print("Please set OPENWEATHER_API_KEY environment variable")
        return
    
    city = input("Enter city name: ").strip()
    if not city:
        print("City name cannot be empty")
        return
    
    weather_json = get_weather_data(city, api_key)
    weather_info = parse_weather_data(weather_json)
    
    if weather_info:
        display_weather_info(weather_info)
        if save_weather_data(weather_info):
            print("Weather data saved successfully")
    else:
        print("Failed to retrieve weather data")

if __name__ == "__main__":
    main()