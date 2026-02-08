import requests
import json
from datetime import datetime

class WeatherAlert:
    def __init__(self, api_key, city, thresholds):
        self.api_key = api_key
        self.city = city
        self.thresholds = thresholds
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"

    def fetch_weather(self):
        params = {
            'q': self.city,
            'appid': self.api_key,
            'units': 'metric'
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None

    def check_alerts(self, weather_data):
        if not weather_data:
            return []
        
        alerts = []
        temp = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        description = weather_data['weather'][0]['description']
        
        if temp > self.thresholds.get('high_temp', 35):
            alerts.append(f"High temperature alert: {temp}°C")
        if temp < self.thresholds.get('low_temp', 5):
            alerts.append(f"Low temperature alert: {temp}°C")
        if humidity > self.thresholds.get('high_humidity', 80):
            alerts.append(f"High humidity alert: {humidity}%")
        
        return alerts

    def send_notification(self, alerts):
        if not alerts:
            print(f"Weather conditions normal for {self.city}")
            return
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n=== Weather Alert for {self.city} ===")
        print(f"Time: {timestamp}")
        for alert in alerts:
            print(f"⚠️  {alert}")
        print("=" * 40)

    def run(self):
        weather_data = self.fetch_weather()
        if weather_data:
            alerts = self.check_alerts(weather_data)
            self.send_notification(alerts)

if __name__ == "__main__":
    config = {
        'api_key': 'your_api_key_here',
        'city': 'London,UK',
        'thresholds': {
            'high_temp': 30,
            'low_temp': 10,
            'high_humidity': 75
        }
    }
    
    alert_system = WeatherAlert(
        config['api_key'],
        config['city'],
        config['thresholds']
    )
    alert_system.run()