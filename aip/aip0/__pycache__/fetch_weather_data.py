import requests
import json
import time
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5/weather"):
        self.api_key = api_key
        self.base_url = base_url
        self.cache = {}
        self.cache_duration = 300

    def get_weather(self, city_name):
        current_time = time.time()
        if city_name in self.cache:
            cached_data, timestamp = self.cache[city_name]
            if current_time - timestamp < self.cache_duration:
                print(f"Returning cached data for {city_name}")
                return cached_data

        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': 'metric'
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['cod'] != 200:
                return {'error': data.get('message', 'Unknown error')}

            processed_data = {
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'weather': data['weather'][0]['main'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'wind_deg': data['wind']['deg'],
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat(),
                'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).isoformat(),
                'sunset': datetime.fromtimestamp(data['sys']['sunset']).isoformat()
            }

            self.cache[city_name] = (processed_data, current_time)
            return processed_data

        except requests.exceptions.RequestException as e:
            return {'error': f'Network error: {str(e)}'}
        except (KeyError, json.JSONDecodeError) as e:
            return {'error': f'Data parsing error: {str(e)}'}

    def get_multiple_cities(self, city_list):
        results = {}
        for city in city_list:
            results[city] = self.get_weather(city)
            time.sleep(0.1)
        return results

    def clear_cache(self):
        self.cache.clear()
        print("Cache cleared")

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris", "Sydney"]
    
    print("Fetching weather data for multiple cities...")
    weather_data = fetcher.get_multiple_cities(cities)
    
    for city, data in weather_data.items():
        if 'error' not in data:
            print(f"\n{city}, {data['country']}:")
            print(f"  Temperature: {data['temperature']}°C (Feels like: {data['feels_like']}°C)")
            print(f"  Conditions: {data['weather']} - {data['description']}")
            print(f"  Humidity: {data['humidity']}% | Pressure: {data['pressure']} hPa")
            print(f"  Wind: {data['wind_speed']} m/s at {data['wind_deg']}°")
            print(f"  Sunrise: {data['sunrise']} | Sunset: {data['sunset']}")
        else:
            print(f"\nError fetching data for {city}: {data['error']}")

if __name__ == "__main__":
    main()