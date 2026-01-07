import requests
import os
from typing import Optional, Dict, Any

def get_current_weather(city: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch current weather data for a given city using OpenWeatherMap API.
    
    Args:
        city: Name of the city to get weather for
        api_key: OpenWeatherMap API key (defaults to WEATHER_API_KEY env var)
    
    Returns:
        Dictionary containing weather data
    
    Raises:
        ValueError: If API key is not provided and not found in environment
        requests.RequestException: If API request fails
    """
    if api_key is None:
        api_key = os.getenv("WEATHER_API_KEY")
        if not api_key:
            raise ValueError("API key must be provided or set as WEATHER_API_KEY environment variable")
    
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }
    
    response = requests.get(base_url, params=params, timeout=10)
    response.raise_for_status()
    
    return response.json()

def display_weather_info(weather_data: Dict[str, Any]) -> None:
    """
    Display formatted weather information.
    
    Args:
        weather_data: Weather data dictionary from OpenWeatherMap API
    """
    if weather_data.get("cod") != 200:
        print(f"Error: {weather_data.get('message', 'Unknown error')}")
        return
    
    main_info = weather_data["main"]
    weather_desc = weather_data["weather"][0]
    
    print(f"Weather in {weather_data['name']}, {weather_data['sys']['country']}:")
    print(f"  Temperature: {main_info['temp']}°C (feels like {main_info['feels_like']}°C)")
    print(f"  Conditions: {weather_desc['description'].title()}")
    print(f"  Humidity: {main_info['humidity']}%")
    print(f"  Pressure: {main_info['pressure']} hPa")
    print(f"  Wind Speed: {weather_data['wind']['speed']} m/s")

if __name__ == "__main__":
    try:
        city_name = input("Enter city name: ").strip()
        if not city_name:
            city_name = "London"
        
        weather = get_current_weather(city_name)
        display_weather_info(weather)
        
    except ValueError as e:
        print(f"Configuration error: {e}")
    except requests.RequestException as e:
        print(f"Network error: {e}")
    except KeyError as e:
        print(f"Unexpected API response format: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")