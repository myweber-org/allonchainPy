
import requests
import os
from typing import Optional, Dict

def get_current_weather(city: str, api_key: Optional[str] = None) -> Dict:
    """
    Fetch current weather data for a given city using OpenWeatherMap API.
    
    Args:
        city: Name of the city to get weather for
        api_key: OpenWeatherMap API key (defaults to WEATHER_API_KEY env var)
    
    Returns:
        Dictionary containing weather data
    
    Raises:
        ValueError: If API key is not provided or city is empty
        requests.exceptions.RequestException: If API request fails
    """
    if not city or not city.strip():
        raise ValueError("City name cannot be empty")
    
    if api_key is None:
        api_key = os.environ.get("WEATHER_API_KEY")
        if not api_key:
            raise ValueError("API key must be provided or set in WEATHER_API_KEY environment variable")
    
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }
    
    response = requests.get(base_url, params=params, timeout=10)
    response.raise_for_status()
    
    return response.json()

def display_weather_data(weather_data: Dict) -> None:
    """
    Display formatted weather information from API response.
    
    Args:
        weather_data: Weather data dictionary from OpenWeatherMap API
    """
    if not weather_data:
        print("No weather data available")
        return
    
    try:
        city = weather_data.get("name", "Unknown")
        country = weather_data.get("sys", {}).get("country", "")
        temp = weather_data.get("main", {}).get("temp")
        humidity = weather_data.get("main", {}).get("humidity")
        description = weather_data.get("weather", [{}])[0].get("description", "Unknown")
        
        location = f"{city}, {country}" if country else city
        
        print(f"Weather in {location}:")
        print(f"  Temperature: {temp}°C")
        print(f"  Humidity: {humidity}%")
        print(f"  Conditions: {description.capitalize()}")
        
    except (KeyError, IndexError, AttributeError) as e:
        print(f"Error parsing weather data: {e}")

if __name__ == "__main__":
    try:
        city_name = input("Enter city name: ").strip()
        weather = get_current_weather(city_name)
        display_weather_data(weather)
    except ValueError as e:
        print(f"Input error: {e}")
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")