
import requests

def fetch_github_user(username):
    """Fetch public information for a given GitHub username."""
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def display_user_info(user_data):
    """Display selected user information in a readable format."""
    if user_data:
        print(f"Username: {user_data.get('login')}")
        print(f"Name: {user_data.get('name', 'Not provided')}")
        print(f"Public Repos: {user_data.get('public_repos')}")
        print(f"Followers: {user_data.get('followers')}")
        print(f"Following: {user_data.get('following')}")
        print(f"Profile URL: {user_data.get('html_url')}")
    else:
        print("User not found or error fetching data.")

if __name__ == "__main__":
    username = input("Enter a GitHub username: ").strip()
    user_info = fetch_github_user(username)
    display_user_info(user_info)