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
            return {'error': f'Failed to fetch weather data: {str(e)}'}
        except KeyError as e:
            return {'error': f'Invalid response format: {str(e)}'}
    
    def _parse_weather_data(self, data):
        return {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'wind_direction': data['wind']['deg'],
            'visibility': data.get('visibility', 'N/A'),
            'cloudiness': data['clouds']['all'],
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def display_weather(self, weather_data):
        if 'error' in weather_data:
            print(f"Error: {weather_data['error']}")
            return
        
        print("\n" + "="*50)
        print(f"Weather Report for {weather_data['city']}, {weather_data['country']}")
        print(f"Generated at: {weather_data['timestamp']}")
        print("="*50)
        print(f"Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)")
        print(f"Weather: {weather_data['weather'].title()}")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Pressure: {weather_data['pressure']} hPa")
        print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
        print(f"Visibility: {weather_data['visibility']} meters")
        print(f"Cloudiness: {weather_data['cloudiness']}%")
        print(f"Sunrise: {weather_data['sunrise']}")
        print(f"Sunset: {weather_data['sunset']}")
        print("="*50)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_weather.py <city_name>")
        print("Example: python fetch_weather.py London")
        sys.exit(1)
    
    city_name = ' '.join(sys.argv[1:])
    
    api_key = "your_api_key_here"
    
    if api_key == "your_api_key_here":
        print("Please replace 'your_api_key_here' with your actual OpenWeatherMap API key")
        print("Get your API key from: https://openweathermap.org/api")
        sys.exit(1)
    
    fetcher = WeatherFetcher(api_key)
    weather_data = fetcher.get_weather(city_name)
    fetcher.display_weather(weather_data)
    
    if 'error' not in weather_data:
        save_option = input("\nSave to file? (y/n): ").lower()
        if save_option == 'y':
            filename = f"weather_{city_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(weather_data, f, indent=2)
            print(f"Weather data saved to {filename}")

if __name__ == "__main__":
    main()