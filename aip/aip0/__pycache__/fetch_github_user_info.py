import requests
import json

def get_github_user_info(username):
    """
    Fetch public information for a given GitHub username.
    """
    url = f"https://api.github.com/users/{username}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        user_data = response.json()
        
        print(f"GitHub User: {user_data.get('login')}")
        print(f"Name: {user_data.get('name', 'Not provided')}")
        print(f"Bio: {user_data.get('bio', 'Not provided')}")
        print(f"Public Repos: {user_data.get('public_repos')}")
        print(f"Followers: {user_data.get('followers')}")
        print(f"Following: {user_data.get('following')}")
        print(f"Profile URL: {user_data.get('html_url')}")
        
        return user_data
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
    return None

if __name__ == "__main__":
    username = input("Enter a GitHub username: ").strip()
    if username:
        get_github_user_info(username)
    else:
        print("No username provided.")