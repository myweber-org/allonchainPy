import requests

def fetch_github_user(username):
    """
    Fetches public information for a given GitHub username.
    """
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
            'html_url': user_data.get('html_url')
        }
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

if __name__ == "__main__":
    username = input("Enter a GitHub username: ")
    user_info = fetch_github_user(username)
    if user_info:
        print(f"\nUser: {user_info['login']}")
        print(f"Name: {user_info['name']}")
        print(f"Public Repos: {user_info['public_repos']}")
        print(f"Followers: {user_info['followers']}")
        print(f"Following: {user_info['following']}")
        print(f"Profile URL: {user_info['html_url']}")
    else:
        print("Could not fetch user details.")