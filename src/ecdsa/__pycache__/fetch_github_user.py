import requests
import sys

def fetch_github_user(username):
    """Fetch GitHub user information from the public API."""
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            print(f"Error: User '{username}' not found on GitHub.")
        else:
            print(f"HTTP error occurred: {e}")
    except requests.exceptions.ConnectionError:
        print("Error: Failed to connect to GitHub API. Check your network.")
    except requests.exceptions.Timeout:
        print("Error: Request to GitHub API timed out.")
    except requests.exceptions.RequestException as e:
        print(f"An unexpected error occurred: {e}")
    return None

def display_user_info(user_data):
    """Display selected user information in a readable format."""
    if not user_data:
        return
    
    print("\n--- GitHub User Information ---")
    print(f"Username: {user_data.get('login')}")
    print(f"Name: {user_data.get('name', 'Not provided')}")
    print(f"Bio: {user_data.get('bio', 'Not provided')}")
    print(f"Public Repos: {user_data.get('public_repos')}")
    print(f"Followers: {user_data.get('followers')}")
    print(f"Following: {user_data.get('following')}")
    print(f"Profile URL: {user_data.get('html_url')}")
    print("-------------------------------")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user.py <github_username>")
        sys.exit(1)
    
    username = sys.argv[1]
    user_data = fetch_github_user(username)
    
    if user_data:
        display_user_info(user_data)