import requests

def get_github_user_info(username):
    """Fetch public information for a given GitHub username."""
    url = f"https://api.github.com/users/{username}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        user_data = response.json()
        
        # Extract and return specific fields
        return {
            "login": user_data.get("login"),
            "name": user_data.get("name"),
            "public_repos": user_data.get("public_repos"),
            "followers": user_data.get("followers"),
            "following": user_data.get("following"),
            "created_at": user_data.get("created_at")
        }
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except Exception as err:
        print(f"An error occurred: {err}")
        return None

if __name__ == "__main__":
    username = input("Enter GitHub username: ").strip()
    if username:
        info = get_github_user_info(username)
        if info:
            print(f"\nUser: {info['login']}")
            print(f"Name: {info['name']}")
            print(f"Public Repositories: {info['public_repos']}")
            print(f"Followers: {info['followers']}")
            print(f"Following: {info['following']}")
            print(f"Account Created: {info['created_at']}")
        else:
            print("Failed to fetch user information.")
    else:
        print("No username provided.")