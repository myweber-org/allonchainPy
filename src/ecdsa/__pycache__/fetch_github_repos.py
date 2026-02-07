import requests
import sys

def get_github_repos(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    
    if response.status_code == 200:
        repos = response.json()
        return [repo['name'] for repo in repos]
    else:
        print(f"Error: Unable to fetch repositories (Status code: {response.status_code})")
        return []

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <github_username>")
        sys.exit(1)
    
    username = sys.argv[1]
    repositories = get_github_repos(username)
    
    if repositories:
        print(f"Public repositories for {username}:")
        for repo in repositories:
            print(f"  - {repo}")
    else:
        print(f"No public repositories found for {username}")