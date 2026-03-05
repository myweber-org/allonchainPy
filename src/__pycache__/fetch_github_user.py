import requests

def fetch_github_user(username):
    """Fetch public information for a given GitHub username."""
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        user_data = response.json()
        return {
            'login': user_data.get('login'),
            'name': user_data.get('name'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following'),
            'created_at': user_data.get('created_at')
        }
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request error occurred: {e}")
        return None

if __name__ == "__main__":
    username = input("Enter a GitHub username: ").strip()
    if username:
        user_info = fetch_github_user(username)
        if user_info:
            print(f"\nUser: {user_info['login']}")
            print(f"Name: {user_info['name']}")
            print(f"Public Repositories: {user_info['public_repos']}")
            print(f"Followers: {user_info['followers']}")
            print(f"Following: {user_info['following']}")
            print(f"Account Created: {user_info['created_at']}")
        else:
            print("Failed to fetch user information.")
    else:
        print("No username provided.")