
import requests
import os

def get_weather(city_name, api_key=None):
    if api_key is None:
        api_key = os.environ.get('OPENWEATHER_API_KEY')
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
            'wind_direction': data['wind'].get('deg', 'N/A')
        }
        
        return weather_info
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error occurred: {str(e)}")
    except (KeyError, ValueError) as e:
        raise Exception(f"Failed to parse weather data: {str(e)}")

def display_weather(weather_data):
    if not weather_data:
        print("No weather data available")
        return
    
    print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
    print(f"  Temperature: {weather_data['temperature']}°C")
    print(f"  Feels like: {weather_data['feels_like']}°C")
    print(f"  Conditions: {weather_data['weather']}")
    print(f"  Humidity: {weather_data['humidity']}%")
    print(f"  Pressure: {weather_data['pressure']} hPa")
    print(f"  Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")

if __name__ == "__main__":
    try:
        city = input("Enter city name: ").strip()
        if not city:
            print("City name cannot be empty")
        else:
            weather = get_weather(city)
            display_weather(weather)
    except Exception as e:
        print(f"Error: {e}")