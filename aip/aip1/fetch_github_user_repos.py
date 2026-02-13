
import requests
import sys

def fetch_repositories(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    if response.status_code == 200:
        repos = response.json()
        return repos
    else:
        print(f"Failed to fetch repositories. Status code: {response.status_code}")
        return None

def display_repositories(repos):
    if not repos:
        print("No repositories found.")
        return
    print(f"Found {len(repos)} repositories:")
    for repo in repos:
        print(f"- {repo['name']}: {repo['description'] or 'No description'}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_user_repos.py <username>")
        sys.exit(1)
    
    username = sys.argv[1]
    repos = fetch_repositories(username)
    if repos:
        display_repositories(repos)