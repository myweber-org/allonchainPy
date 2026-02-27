import requests

def fetch_github_user(username):
    """
    Fetch public profile information for a given GitHub username.
    Returns a dictionary containing user data or None if user not found.
    """
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            print(f"User '{username}' not found on GitHub.")
        else:
            print(f"HTTP error occurred: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    return None

def display_user_info(user_data):
    """Display selected user information in a readable format."""
    if not user_data:
        return
    
    print(f"GitHub Profile: {user_data.get('html_url')}")
    print(f"Username: {user_data.get('login')}")
    print(f"Name: {user_data.get('name', 'Not provided')}")
    print(f"Bio: {user_data.get('bio', 'Not provided')}")
    print(f"Public Repositories: {user_data.get('public_repos')}")
    print(f"Followers: {user_data.get('followers')}")
    print(f"Following: {user_data.get('following')}")
    print(f"Account Created: {user_data.get('created_at')}")

if __name__ == "__main__":
    username = input("Enter GitHub username: ").strip()
    if username:
        user_data = fetch_github_user(username)
        display_user_info(user_data)