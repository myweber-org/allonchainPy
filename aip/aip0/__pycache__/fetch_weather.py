import requests
import json
from datetime import datetime
import logging

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5/weather"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def get_weather_by_city(self, city_name, country_code=None):
        query = city_name
        if country_code:
            query += f",{country_code}"
        
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('cod') != 200:
                logging.error(f"API error: {data.get('message', 'Unknown error')}")
                return None
            
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {e}")
            return None

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
            'wind_direction': data['wind'].get('deg', 'N/A'),
            'visibility': data.get('visibility', 'N/A'),
            'cloudiness': data['clouds']['all'],
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
    API_KEY = "your_api_key_here"
    
    fetcher = WeatherFetcher(API_KEY)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        print(f"\nFetching weather for {city}...")
        weather = fetcher.get_weather_by_city(city)
        if weather:
            fetcher.display_weather(weather)
        else:
            print(f"Failed to fetch weather for {city}")

if __name__ == "__main__":
    main()
import requests
import json
from datetime import datetime
import sys

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5/weather"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        
    def get_weather_by_city(self, city_name, units="metric"):
        params = {
            "q": city_name,
            "appid": self.api_key,
            "units": units
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return self._parse_weather_data(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
        except json.JSONDecodeError:
            print("Error parsing API response")
            return None
    
    def _parse_weather_data(self, data):
        if data.get("cod") != 200:
            print(f"API error: {data.get('message', 'Unknown error')}")
            return None
            
        weather_info = {
            "city": data.get("name"),
            "country": data.get("sys", {}).get("country"),
            "temperature": data.get("main", {}).get("temp"),
            "feels_like": data.get("main", {}).get("feels_like"),
            "humidity": data.get("main", {}).get("humidity"),
            "pressure": data.get("main", {}).get("pressure"),
            "weather": data.get("weather", [{}])[0].get("description"),
            "wind_speed": data.get("wind", {}).get("speed"),
            "wind_direction": data.get("wind", {}).get("deg"),
            "visibility": data.get("visibility"),
            "clouds": data.get("clouds", {}).get("all"),
            "sunrise": datetime.fromtimestamp(data.get("sys", {}).get("sunrise")).strftime('%H:%M:%S'),
            "sunset": datetime.fromtimestamp(data.get("sys", {}).get("sunset")).strftime('%H:%M:%S'),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return weather_info
    
    def display_weather(self, weather_data):
        if not weather_data:
            print("No weather data available")
            return
            
        print("\n" + "="*50)
        print(f"Weather Report for {weather_data['city']}, {weather_data['country']}")
        print("="*50)
        print(f"Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)")
        print(f"Weather: {weather_data['weather'].title()}")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Pressure: {weather_data['pressure']} hPa")
        print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
        print(f"Visibility: {weather_data['visibility']} meters")
        print(f"Cloud Cover: {weather_data['clouds']}%")
        print(f"Sunrise: {weather_data['sunrise']}")
        print(f"Sunset: {weather_data['sunset']}")
        print(f"Report Time: {weather_data['timestamp']}")
        print("="*50)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_weather.py <city_name>")
        print("Example: python fetch_weather.py London")
        sys.exit(1)
    
    city_name = " ".join(sys.argv[1:])
    
    api_key = "your_api_key_here"
    
    fetcher = WeatherFetcher(api_key)
    weather_data = fetcher.get_weather_by_city(city_name)
    
    if weather_data:
        fetcher.display_weather(weather_data)
        
        with open(f"weather_{city_name.replace(' ', '_')}.json", "w") as f:
            json.dump(weather_data, f, indent=2)
        print(f"\nWeather data saved to weather_{city_name.replace(' ', '_')}.json")
    else:
        print(f"Failed to fetch weather data for {city_name}")

if __name__ == "__main__":
    main()