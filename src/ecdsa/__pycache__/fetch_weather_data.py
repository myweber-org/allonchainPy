
import requests
import json
import os

def get_weather(city_name, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}q={city_name}&appid={api_key}&units=metric"
    
    response = requests.get(complete_url)
    data = response.json()
    
    if data["cod"] != "404":
        main = data["main"]
        weather_desc = data["weather"][0]["description"]
        temperature = main["temp"]
        pressure = main["pressure"]
        humidity = main["humidity"]
        
        print(f"Weather in {city_name}:")
        print(f"Description: {weather_desc}")
        print(f"Temperature: {temperature}°C")
        print(f"Pressure: {pressure} hPa")
        print(f"Humidity: {humidity}%")
    else:
        print("City not found.")

if __name__ == "__main__":
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        print("Please set OPENWEATHER_API_KEY environment variable.")
        exit(1)
    
    city = input("Enter city name: ")
    get_weather(city, api_key)