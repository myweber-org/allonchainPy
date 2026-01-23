import requests
import json
from datetime import datetime

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
            'wind_deg': data['wind']['deg'],
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S'),
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S')
        }
        
        return weather_info
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing weather data: {e}")
        return None

def display_weather(weather_data):
    if not weather_data:
        print("No weather data available.")
        return
    
    print("\n" + "="*50)
    print(f"Weather in {weather_data['city']}, {weather_data['country']}")
    print("="*50)
    print(f"Current Time: {weather_data['timestamp']}")
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels Like: {weather_data['feels_like']}°C")
    print(f"Weather: {weather_data['weather']} - {weather_data['description']}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_deg']}°")
    print(f"Sunrise: {weather_data['sunrise']}")
    print(f"Sunset: {weather_data['sunset']}")
    print("="*50)

def main():
    api_key = "your_api_key_here"
    city = "London"
    
    print(f"Fetching weather data for {city}...")
    weather_data = get_weather(api_key, city)
    
    if weather_data:
        display_weather(weather_data)
        
        with open('weather_data.json', 'w') as f:
            json.dump(weather_data, f, indent=2)
        print("\nWeather data saved to 'weather_data.json'")
    else:
        print("Failed to fetch weather data.")

if __name__ == "__main__":
    main()