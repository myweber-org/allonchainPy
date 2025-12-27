import requests
import json
from datetime import datetime

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
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return {
                'city': data['name'],
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'],
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
            }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def save_to_file(self, weather_data, filename='weather_data.json'):
        if weather_data:
            try:
                with open(filename, 'a') as f:
                    json.dump(weather_data, f)
                    f.write('\n')
                print(f"Weather data saved to {filename}")
            except IOError as e:
                print(f"Error saving to file: {e}")

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ['London', 'New York', 'Tokyo', 'Paris']
    
    for city in cities:
        print(f"Fetching weather for {city}...")
        weather = fetcher.get_weather(city)
        
        if weather:
            print(f"Temperature in {weather['city']}: {weather['temperature']}°C")
            print(f"Conditions: {weather['description']}")
            fetcher.save_to_file(weather)
        print("-" * 40)

if __name__ == "__main__":
    main()