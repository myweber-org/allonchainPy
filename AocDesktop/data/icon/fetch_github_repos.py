import requests
import sys

def fetch_github_repos(username):
    """Fetch public repositories for a given GitHub username."""
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    
    if response.status_code == 200:
        repos = response.json()
        repo_names = [repo['name'] for repo in repos]
        return repo_names
    else:
        print(f"Error: Unable to fetch repositories. Status code: {response.status_code}")
        return []

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <github_username>")
        sys.exit(1)
    
    username = sys.argv[1]
    repos = fetch_github_repos(username)
    
    if repos:
        print(f"Public repositories for user '{username}':")
        for repo in repos:
            print(f"  - {repo}")
    else:
        print(f"No public repositories found for user '{username}'.")