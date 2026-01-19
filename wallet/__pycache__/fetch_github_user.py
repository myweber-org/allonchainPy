import requests

def get_github_user(username):
    """
    Fetch user information from the GitHub API.
    """
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    
    if response.status_code == 200:
        user_data = response.json()
        return {
            'name': user_data.get('name'),
            'login': user_data.get('login'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following'),
            'created_at': user_data.get('created_at')
        }
    else:
        return None

def display_user_info(user_info):
    """
    Display the fetched user information in a formatted way.
    """
    if user_info:
        print(f"Name: {user_info['name']}")
        print(f"Username: {user_info['login']}")
        print(f"Public Repositories: {user_info['public_repos']}")
        print(f"Followers: {user_info['followers']}")
        print(f"Following: {user_info['following']}")
        print(f"Account Created: {user_info['created_at']}")
    else:
        print("User not found or error fetching data.")

if __name__ == "__main__":
    username = input("Enter a GitHub username: ")
    user_info = get_github_user(username)
    display_user_info(user_info)