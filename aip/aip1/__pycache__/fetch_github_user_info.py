import requests

def get_github_user_info(username):
    """
    Fetches public information for a given GitHub username.
    """
    url = f"https://api.github.com/users/{username}"
    headers = {'Accept': 'application/vnd.github.v3+json'}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        user_data = response.json()

        info = {
            'login': user_data.get('login'),
            'name': user_data.get('name'),
            'company': user_data.get('company'),
            'blog': user_data.get('blog'),
            'location': user_data.get('location'),
            'email': user_data.get('email'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following'),
            'created_at': user_data.get('created_at')
        }
        return info
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except Exception as err:
        print(f"An error occurred: {err}")
        return None

if __name__ == "__main__":
    username = input("Enter a GitHub username: ").strip()
    if username:
        user_info = get_github_user_info(username)
        if user_info:
            print("\nGitHub User Information:")
            for key, value in user_info.items():
                print(f"{key.replace('_', ' ').title()}: {value}")
        else:
            print("Failed to fetch user information.")
    else:
        print("No username provided.")