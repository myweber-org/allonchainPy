
import requests

def get_github_user(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        return None

def display_user_info(user_data):
    if user_data:
        print(f"Username: {user_data.get('login')}")
        print(f"Name: {user_data.get('name')}")
        print(f"Followers: {user_data.get('followers')}")
        print(f"Public Repos: {user_data.get('public_repos')}")
        print(f"Location: {user_data.get('location')}")
        print(f"Bio: {user_data.get('bio')}")
    else:
        print("User not found or error occurred.")

if __name__ == "__main__":
    username = input("Enter GitHub username: ")
    user_info = get_github_user(username)
    display_user_info(user_info)
import requests
import sys

def get_github_user(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    
    if response.status_code == 200:
        user_data = response.json()
        print(f"Username: {user_data['login']}")
        print(f"Name: {user_data.get('name', 'N/A')}")
        print(f"Public Repos: {user_data['public_repos']}")
        print(f"Followers: {user_data['followers']}")
        print(f"Following: {user_data['following']}")
        print(f"Profile URL: {user_data['html_url']}")
    else:
        print(f"Error: User '{username}' not found or API request failed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user.py <username>")
        sys.exit(1)
    
    username = sys.argv[1]
    get_github_user(username)