
import requests
import os

def get_weather(city_name, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}q={city_name}&appid={api_key}&units=metric"
    response = requests.get(complete_url)
    data = response.json()
    if data["cod"] != "404":
        main = data["main"]
        temperature = main["temp"]
        pressure = main["pressure"]
        humidity = main["humidity"]
        weather_desc = data["weather"][0]["description"]
        print(f"Temperature: {temperature}°C")
        print(f"Atmospheric Pressure: {pressure} hPa")
        print(f"Humidity: {humidity}%")
        print(f"Weather Description: {weather_desc}")
    else:
        print("City not found.")

if __name__ == "__main__":
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        print("Please set the OPENWEATHER_API_KEY environment variable.")
    else:
        city = input("Enter city name: ")
        get_weather(city, api_key)