
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

class WeatherAlert:
    def __init__(self, config_file='config.json'):
        self.config = self.load_config(config_file)
        self.alert_history = []
        
    def load_config(self, config_file):
        default_config = {
            "temperature_thresholds": {
                "high": 35.0,
                "low": 5.0
            },
            "notification_emails": ["admin@example.com"],
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "email_credentials": {
                "username": "your_email@gmail.com",
                "password": "your_password"
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return default_config
    
    def check_temperature(self, current_temp, location="Unknown"):
        alerts = []
        
        if current_temp > self.config["temperature_thresholds"]["high"]:
            alert_msg = f"Heat alert! Temperature {current_temp}°C exceeds high threshold of {self.config['temperature_thresholds']['high']}°C in {location}"
            alerts.append(("HEAT", alert_msg))
            
        if current_temp < self.config["temperature_thresholds"]["low"]:
            alert_msg = f"Cold alert! Temperature {current_temp}°C below low threshold of {self.config['temperature_thresholds']['low']}°C in {location}"
            alerts.append(("COLD", alert_msg))
        
        if alerts:
            self.trigger_alerts(alerts, location, current_temp)
        
        return alerts
    
    def trigger_alerts(self, alerts, location, temperature):
        timestamp = datetime.now().isoformat()
        
        for alert_type, message in alerts:
            alert_record = {
                "timestamp": timestamp,
                "type": alert_type,
                "location": location,
                "temperature": temperature,
                "message": message
            }
            
            self.alert_history.append(alert_record)
            self.send_email_alert(message)
            self.log_alert(alert_record)
    
    def send_email_alert(self, message):
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config["email_credentials"]["username"]
            msg['To'] = ", ".join(self.config["notification_emails"])
            msg['Subject'] = "Weather Alert Notification"
            
            body = f"Weather Alert:\n\n{message}\n\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config["smtp_server"], self.config["smtp_port"])
            server.starttls()
            server.login(self.config["email_credentials"]["username"],
                        self.config["email_credentials"]["password"])
            server.send_message(msg)
            server.quit()
            
            return True
        except Exception as e:
            print(f"Failed to send email: {e}")
            return False
    
    def log_alert(self, alert_record):
        log_file = "weather_alerts.log"
        with open(log_file, 'a') as f:
            f.write(json.dumps(alert_record) + '\n')
    
    def get_alert_history(self, alert_type=None):
        if alert_type:
            return [alert for alert in self.alert_history if alert["type"] == alert_type]
        return self.alert_history
    
    def clear_history(self):
        self.alert_history = []

def main():
    alert_system = WeatherAlert()
    
    test_temperatures = [
        (40.5, "Tokyo"),
        (3.2, "London"),
        (25.0, "New York"),
        (-2.0, "Moscow"),
        (36.0, "Dubai")
    ]
    
    for temp, location in test_temperatures:
        alerts = alert_system.check_temperature(temp, location)
        if alerts:
            print(f"Alerts triggered for {location}:")
            for alert in alerts:
                print(f"  - {alert[1]}")
        else:
            print(f"No alerts for {location} at {temp}°C")
    
    print(f"\nTotal alerts triggered: {len(alert_system.get_alert_history())}")

if __name__ == "__main__":
    main()