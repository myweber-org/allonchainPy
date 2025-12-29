
import requests

def fetch_github_user(username):
    """Fetch and display basic information for a GitHub user."""
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        user_data = response.json()
        
        print(f"Username: {user_data.get('login')}")
        print(f"Name: {user_data.get('name', 'Not provided')}")
        print(f"Public Repos: {user_data.get('public_repos')}")
        print(f"Followers: {user_data.get('followers')}")
        print(f"Following: {user_data.get('following')}")
        print(f"Profile URL: {user_data.get('html_url')}")
        
        return user_data
    except requests.exceptions.HTTPError as e:
        print(f"Error fetching user: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

if __name__ == "__main__":
    fetch_github_user("octocat")