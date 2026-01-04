import requests
import json
import os
from datetime import datetime

def get_weather(city_name, api_key=None):
    """
    Fetch current weather data for a given city using OpenWeatherMap API.
    """
    if api_key is None:
        api_key = os.environ.get('OPENWEATHER_API_KEY')
        if api_key is None:
            raise ValueError("API key not provided and OPENWEATHER_API_KEY environment variable not set.")

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

        if data.get('cod') != 200:
            raise Exception(f"API Error: {data.get('message', 'Unknown error')}")

        weather_info = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'wind_deg': data['wind']['deg'],
            'visibility': data.get('visibility', 'N/A'),
            'clouds': data['clouds']['all'],
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }
        return weather_info

    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error occurred: {e}")
    except (KeyError, IndexError) as e:
        raise Exception(f"Unexpected API response format: {e}")

def display_weather(weather_data):
    """
    Display weather information in a readable format.
    """
    if not weather_data:
        print("No weather data to display.")
        return

    print("\n" + "="*40)
    print(f"Weather in {weather_data['city']}, {weather_data['country']}")
    print(f"Report Time: {weather_data['timestamp']}")
    print("="*40)
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels Like: {weather_data['feels_like']}°C")
    print(f"Weather: {weather_data['weather'].title()}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_deg']}°")
    print(f"Visibility: {weather_data['visibility']} meters")
    print(f"Cloud Cover: {weather_data['clouds']}%")
    print(f"Sunrise: {weather_data['sunrise']}")
    print(f"Sunset: {weather_data['sunset']}")
    print("="*40)

if __name__ == "__main__":
    try:
        city = input("Enter city name: ").strip()
        if not city:
            print("City name cannot be empty.")
        else:
            weather = get_weather(city)
            display_weather(weather)
    except Exception as e:
        print(f"Error: {e}")