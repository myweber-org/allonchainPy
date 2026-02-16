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
            'wind_speed': data['wind']['speed']
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
        
    print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
    print(f"  Temperature: {weather_data['temperature']}°C")
    print(f"  Feels like: {weather_data['feels_like']}°C")
    print(f"  Humidity: {weather_data['humidity']}%")
    print(f"  Pressure: {weather_data['pressure']} hPa")
    print(f"  Conditions: {weather_data['weather'].title()}")
    print(f"  Wind Speed: {weather_data['wind_speed']} m/s")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather_data.py <api_key> <city>")
        print("Example: python fetch_weather_data.py abc123 London")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])
    
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)import requests
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
    except json.JSONDecodeError:
        print("Error decoding JSON response.")
        return None

def display_weather(data):
    if data and data.get('cod') == 200:
        main = data['main']
        weather = data['weather'][0]
        print(f"City: {data['name']}")
        print(f"Temperature: {main['temp']}°C")
        print(f"Humidity: {main['humidity']}%")
        print(f"Weather: {weather['description']}")
        print(f"Wind Speed: {data['wind']['speed']} m/s")
    else:
        error_msg = data.get('message', 'Unknown error') if data else 'No data received'
        print(f"Failed to get weather data: {error_msg}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather_data.py <API_KEY> <CITY_NAME>")
        sys.exit(1)

    api_key = sys.argv[1]
    city = sys.argv[2]
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)import requests
import json
from datetime import datetime
import logging

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def get_current_weather(self, city_name, country_code=None):
        query = f"{city_name},{country_code}" if country_code else city_name
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(f"{self.base_url}/weather", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('cod') != 200:
                logging.error(f"API error: {data.get('message', 'Unknown error')}")
                return None
            
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON response: {e}")
            return None

    def _parse_weather_data(self, raw_data):
        return {
            'city': raw_data['name'],
            'country': raw_data['sys']['country'],
            'temperature': raw_data['main']['temp'],
            'feels_like': raw_data['main']['feels_like'],
            'humidity': raw_data['main']['humidity'],
            'pressure': raw_data['main']['pressure'],
            'weather': raw_data['weather'][0]['description'],
            'wind_speed': raw_data['wind']['speed'],
            'wind_direction': raw_data['wind'].get('deg', 0),
            'visibility': raw_data.get('visibility', 0),
            'clouds': raw_data['clouds']['all'],
            'sunrise': datetime.fromtimestamp(raw_data['sys']['sunrise']).isoformat(),
            'sunset': datetime.fromtimestamp(raw_data['sys']['sunset']).isoformat(),
            'timestamp': datetime.now().isoformat()
        }

    def save_to_file(self, weather_data, filename="weather_data.json"):
        if not weather_data:
            logging.warning("No data to save")
            return False
        
        try:
            with open(filename, 'a') as f:
                json.dump(weather_data, f, indent=2)
                f.write('\n')
            logging.info(f"Weather data saved to {filename}")
            return True
        except IOError as e:
            logging.error(f"Failed to save data: {e}")
            return False

def main():
    api_key = "your_api_key_here"  # Replace with actual API key
    fetcher = WeatherFetcher(api_key)
    
    cities = [
        ("London", "GB"),
        ("New York", "US"),
        ("Tokyo", "JP"),
        ("Sydney", "AU")
    ]
    
    for city, country in cities:
        logging.info(f"Fetching weather for {city}, {country}")
        weather_data = fetcher.get_current_weather(city, country)
        
        if weather_data:
            print(f"\nWeather in {weather_data['city']}, {weather_data['country']}:")
            print(f"Temperature: {weather_data['temperature']}°C")
            print(f"Feels like: {weather_data['feels_like']}°C")
            print(f"Weather: {weather_data['weather']}")
            print(f"Humidity: {weather_data['humidity']}%")
            print(f"Wind: {weather_data['wind_speed']} m/s")
            
            fetcher.save_to_file(weather_data)
        else:
            print(f"Failed to fetch weather for {city}, {country}")

if __name__ == "__main__":
    main()import requests
import json
import os
from datetime import datetime

def get_weather(city_name, api_key=None):
    """
    Fetches current weather data for a given city from OpenWeatherMap API.
    
    Args:
        city_name (str): Name of the city to get weather for
        api_key (str, optional): OpenWeatherMap API key. If not provided,
                                 tries to get from environment variable.
    
    Returns:
        dict: Weather data if successful, None otherwise
    """
    if api_key is None:
        api_key = os.getenv('OPENWEATHER_API_KEY')
    
    if not api_key:
        print("Error: API key not provided and OPENWEATHER_API_KEY environment variable not set.")
        return None
    
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'  # Use metric units (Celsius)
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('cod') != 200:
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None
        
        # Extract relevant weather information
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
        
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error parsing response: {e}")
        return None

def display_weather(weather_data):
    """
    Displays weather information in a readable format.
    
    Args:
        weather_data (dict): Weather data dictionary returned by get_weather()
    """
    if not weather_data:
        print("No weather data to display.")
        return
    
    print("\n" + "="*50)
    print(f"Weather in {weather_data['city']}, {weather_data['country']}")
    print("="*50)
    print(f"Current Time: {weather_data['timestamp']}")
    print(f"Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)")
    print(f"Weather: {weather_data['weather']} - {weather_data['description']}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
    print(f"Cloudiness: {weather_data['cloudiness']}%")
    if weather_data['visibility'] != 'N/A':
        print(f"Visibility: {weather_data['visibility'] / 1000:.1f} km")
    print(f"Sunrise: {weather_data['sunrise']}")
    print(f"Sunset: {weather_data['sunset']}")
    print("="*50)

if __name__ == "__main__":
    # Example usage
    city = "London"
    print(f"Fetching weather data for {city}...")
    
    # Note: In production, set OPENWEATHER_API_KEY environment variable
    # or pass your API key directly
    weather = get_weather(city)
    
    if weather:
        display_weather(weather)
    else:
        print("Failed to retrieve weather data.")