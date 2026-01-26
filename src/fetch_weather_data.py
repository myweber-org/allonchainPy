
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
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(weather_data):
    if weather_data and weather_data.get('cod') == 200:
        main = weather_data['main']
        weather = weather_data['weather'][0]
        print(f"City: {weather_data['name']}")
        print(f"Temperature: {main['temp']}°C")
        print(f"Humidity: {main['humidity']}%")
        print(f"Conditions: {weather['description'].capitalize()}")
        print(f"Pressure: {main['pressure']} hPa")
    else:
        print("City not found or invalid data received.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather_data.py <api_key> <city_name>")
        sys.exit(1)

    api_key = sys.argv[1]
    city = sys.argv[2]
    weather_info = get_weather(api_key, city)
    display_weather(weather_info)import requests
import json
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5/weather"):
        self.api_key = api_key
        self.base_url = base_url

    def get_weather_by_city(self, city_name, units="metric"):
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': units
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return self._parse_weather_data(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None

    def _parse_weather_data(self, data):
        if data.get('cod') != 200:
            return None
        
        weather_info = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['main'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'wind_direction': data['wind'].get('deg', 'N/A'),
            'visibility': data.get('visibility', 'N/A'),
            'cloudiness': data['clouds']['all'],
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }
        return weather_info

    def display_weather(self, weather_data):
        if not weather_data:
            print("No weather data available.")
            return
        
        print("\n" + "="*50)
        print(f"Weather Report for {weather_data['city']}, {weather_data['country']}")
        print("="*50)
        print(f"Current Time: {weather_data['timestamp']}")
        print(f"Weather: {weather_data['weather']} ({weather_data['description']})")
        print(f"Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Pressure: {weather_data['pressure']} hPa")
        print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
        print(f"Visibility: {weather_data['visibility']} meters")
        print(f"Cloudiness: {weather_data['cloudiness']}%")
        print(f"Sunrise: {weather_data['sunrise']}")
        print(f"Sunset: {weather_data['sunset']}")
        print("="*50)

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris", "Sydney"]
    
    for city in cities:
        print(f"\nFetching weather for {city}...")
        weather_data = fetcher.get_weather_by_city(city)
        if weather_data:
            fetcher.display_weather(weather_data)
        else:
            print(f"Could not fetch weather data for {city}")

if __name__ == "__main__":
    main()