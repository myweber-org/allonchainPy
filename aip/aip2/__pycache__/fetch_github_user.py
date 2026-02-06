
import requests

def get_github_user(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to fetch user: {response.status_code}"}

if __name__ == "__main__":
    user_data = get_github_user("octocat")
    print(user_data)
import requests

def get_github_user_info(username):
    """
    Fetches public information for a given GitHub username.
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
            'blog': user_data.get('blog'),
            'location': user_data.get('location'),
            'bio': user_data.get('bio')
        }
    else:
        print(f"Error: Unable to fetch data for user '{username}'. Status code: {response.status_code}")
        return None

def display_user_info(user_info):
    """
    Prints the user information in a formatted way.
    """
    if user_info:
        print(f"GitHub User: {user_info['login']}")
        print(f"Name: {user_info['name']}")
        print(f"Bio: {user_info['bio']}")
        print(f"Location: {user_info['location']}")
        print(f"Public Repositories: {user_info['public_repos']}")
        print(f"Followers: {user_info['followers']}")
        print(f"Following: {user_info['following']}")
        print(f"Blog/Website: {user_info['blog']}")
    else:
        print("No user information to display.")

if __name__ == "__main__":
    username = input("Enter a GitHub username: ").strip()
    if username:
        info = get_github_user_info(username)
        display_user_info(info)
    else:
        print("No username provided.")