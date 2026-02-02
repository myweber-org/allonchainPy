
import requests
import sys

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
            'public_repos': user_data.get('public_repos', 0),
            'followers': user_data.get('followers', 0),
            'following': user_data.get('following', 0),
            'location': user_data.get('location'),
            'blog': user_data.get('blog')
        }
    except requests.exceptions.HTTPError as e:
        print(f"Error fetching user '{username}': {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user.py <username>")
        sys.exit(1)

    username = sys.argv[1]
    user_info = fetch_github_user(username)

    if user_info:
        print(f"GitHub User: {user_info['login']}")
        if user_info['name']:
            print(f"Name: {user_info['name']}")
        print(f"Public Repositories: {user_info['public_repos']}")
        print(f"Followers: {user_info['followers']}")
        print(f"Following: {user_info['following']}")
        if user_info['location']:
            print(f"Location: {user_info['location']}")
        if user_info['blog']:
            print(f"Blog/Website: {user_info['blog']}")

if __name__ == "__main__":
    main()