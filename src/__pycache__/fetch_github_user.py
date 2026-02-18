import requests
import json

def fetch_github_user(username):
    """
    Fetches public profile information for a given GitHub username.
    """
    url = f"https://api.github.com/users/{username}"
    headers = {'Accept': 'application/vnd.github.v3+json'}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        user_data = response.json()
        return user_data
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
    return None

def display_user_info(user_data):
    """
    Prints selected information from the user's profile data.
    """
    if user_data:
        print(f"Username: {user_data.get('login')}")
        print(f"Name: {user_data.get('name')}")
        print(f"Bio: {user_data.get('bio')}")
        print(f"Public Repos: {user_data.get('public_repos')}")
        print(f"Followers: {user_data.get('followers')}")
        print(f"Following: {user_data.get('following')}")
        print(f"Profile URL: {user_data.get('html_url')}")
    else:
        print("No user data to display.")

if __name__ == "__main__":
    username = input("Enter a GitHub username: ").strip()
    if username:
        data = fetch_github_user(username)
        display_user_info(data)
    else:
        print("No username provided.")
import requests

def fetch_github_user(username):
    """
    Fetches public profile information for a given GitHub username.
    """
    url = f"https://api.github.com/users/{username}"
    headers = {"Accept": "application/vnd.github.v3+json"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        user_data = response.json()
        return {
            "login": user_data.get("login"),
            "name": user_data.get("name"),
            "public_repos": user_data.get("public_repos"),
            "followers": user_data.get("followers"),
            "following": user_data.get("following"),
            "html_url": user_data.get("html_url")
        }
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            return {"error": f"User '{username}' not found."}
        else:
            return {"error": f"HTTP error occurred: {e}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}

if __name__ == "__main__":
    username = input("Enter a GitHub username: ").strip()
    result = fetch_github_user(username)
    print(result)