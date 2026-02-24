
import requests
import sys

def fetch_github_user(username):
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        user_data = response.json()
        print(f"Username: {user_data.get('login')}")
        print(f"Name: {user_data.get('name')}")
        print(f"Public Repos: {user_data.get('public_repos')}")
        print(f"Followers: {user_data.get('followers')}")
        print(f"Following: {user_data.get('following')}")
        print(f"Profile URL: {user_data.get('html_url')}")
    except requests.exceptions.HTTPError as e:
        print(f"Error fetching user: {e}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user.py <username>")
        sys.exit(1)
    fetch_github_user(sys.argv[1])import requests

def fetch_github_user(username):
    """Fetch public details of a GitHub user."""
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to fetch user {username}. Status code: {response.status_code}"}

if __name__ == "__main__":
    user_data = fetch_github_user("octocat")
    print(user_data)import requests
import json

def get_github_user_info(username):
    """Fetch public information for a given GitHub username."""
    url = f"https://api.github.com/users/{username}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        user_data = response.json()
        
        return {
            "login": user_data.get("login"),
            "name": user_data.get("name"),
            "public_repos": user_data.get("public_repos", 0),
            "followers": user_data.get("followers", 0),
            "following": user_data.get("following", 0),
            "created_at": user_data.get("created_at")
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def display_user_info(user_info):
    """Display formatted user information."""
    if not user_info:
        print("No user data to display.")
        return
    
    print("\nGitHub User Information")
    print("=" * 30)
    print(f"Username: {user_info['login']}")
    print(f"Name: {user_info['name'] or 'Not provided'}")
    print(f"Public Repositories: {user_info['public_repos']}")
    print(f"Followers: {user_info['followers']}")
    print(f"Following: {user_info['following']}")
    print(f"Account Created: {user_info['created_at']}")

if __name__ == "__main__":
    username = input("Enter GitHub username: ").strip()
    if username:
        user_info = get_github_user_info(username)
        display_user_info(user_info)
    else:
        print("No username provided.")