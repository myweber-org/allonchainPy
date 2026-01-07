
import requests
import json
import os
from datetime import datetime

def get_weather_data(city_name, api_key=None):
    """
    Fetch current weather data for a given city using OpenWeatherMap API.
    """
    if api_key is None:
        api_key = os.getenv('OPENWEATHER_API_KEY')
        if api_key is None:
            raise ValueError("API key not provided and OPENWEATHER_API_KEY environment variable not set")

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
            error_message = data.get('message', 'Unknown error')
            raise Exception(f"API Error: {error_message}")

        return {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'wind_direction': data['wind'].get('deg', 'N/A'),
            'visibility': data.get('visibility', 'N/A'),
            'cloudiness': data['clouds']['all'],
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }

    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error occurred: {str(e)}")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise Exception(f"Error parsing API response: {str(e)}")

def display_weather_data(weather_info):
    """
    Display weather information in a readable format.
    """
    if not weather_info:
        print("No weather data to display.")
        return

    print("\n" + "="*50)
    print(f"Weather Report for {weather_info['city']}, {weather_info['country']}")
    print(f"Report Time: {weather_info['timestamp']}")
    print("="*50)
    print(f"Temperature: {weather_info['temperature']}°C (Feels like: {weather_info['feels_like']}°C)")
    print(f"Weather: {weather_info['weather'].title()}")
    print(f"Humidity: {weather_info['humidity']}%")
    print(f"Pressure: {weather_info['pressure']} hPa")
    print(f"Wind: {weather_info['wind_speed']} m/s at {weather_info['wind_direction']}°")
    print(f"Visibility: {weather_info['visibility']} meters")
    print(f"Cloudiness: {weather_info['cloudiness']}%")
    print(f"Sunrise: {weather_info['sunrise']}")
    print(f"Sunset: {weather_info['sunset']}")
    print("="*50)

if __name__ == "__main__":
    try:
        city = input("Enter city name: ").strip()
        if not city:
            print("City name cannot be empty.")
            exit(1)

        weather_data = get_weather_data(city)
        display_weather_data(weather_data)

    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"Error: {e}")