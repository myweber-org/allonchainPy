import requests
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
        return {
            'city': data['name'],
            'temperature': data['main']['temp'],
            'description': data['weather'][0]['description'],
            'humidity': data['main']['humidity']
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    city_name = input("Enter city name: ")
    weather_info = get_weather(API_KEY, city_name)
    if weather_info:
        print(f"Weather in {weather_info['city']}:")
        print(f"Temperature: {weather_info['temperature']}°C")
        print(f"Condition: {weather_info['description']}")
        print(f"Humidity: {weather_info['humidity']}%")
import requests
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeatherFetcher:
    def __init__(self, api_key: str, base_url: str = "http://api.openweathermap.org/data/2.5/weather"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'WeatherFetcher/1.0'})

    def get_weather(self, city: str, country_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        query = city
        if country_code:
            query = f"{city},{country_code}"
        
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            logger.info(f"Fetching weather data for {query}")
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            processed_data = {
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
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully fetched weather data for {processed_data['city']}")
            return processed_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error occurred: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return None
        except KeyError as e:
            logger.error(f"Unexpected data structure in response: {e}")
            return None

    def save_to_file(self, data: Dict[str, Any], filename: str = "weather_data.json"):
        try:
            with open(filename, 'a') as f:
                json.dump(data, f, indent=2)
                f.write('\n')
            logger.info(f"Weather data saved to {filename}")
        except IOError as e:
            logger.error(f"Failed to save data to file: {e}")

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = [
        ("London", "UK"),
        ("New York", "US"),
        ("Tokyo", "JP"),
        ("Sydney", "AU")
    ]
    
    for city, country in cities:
        weather_data = fetcher.get_weather(city, country)
        if weather_data:
            print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
            print(f"  Temperature: {weather_data['temperature']}°C")
            print(f"  Feels like: {weather_data['feels_like']}°C")
            print(f"  Conditions: {weather_data['weather']}")
            print(f"  Humidity: {weather_data['humidity']}%")
            print(f"  Wind: {weather_data['wind_speed']} m/s")
            print("-" * 40)
            
            fetcher.save_to_file(weather_data)
        else:
            print(f"Failed to fetch weather data for {city}, {country}")

if __name__ == "__main__":
    main()