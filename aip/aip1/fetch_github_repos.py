import requests
import sys

def fetch_user_repos(username, per_page=30):
    repos = []
    page = 1
    while True:
        url = f"https://api.github.com/users/{username}/repos"
        params = {"per_page": per_page, "page": page}
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error: Unable to fetch repositories (Status: {response.status_code})")
            sys.exit(1)
        
        data = response.json()
        if not data:
            break
            
        repos.extend(data)
        page += 1
        
    return repos

def display_repos(repos):
    for idx, repo in enumerate(repos, 1):
        print(f"{idx}. {repo['name']}")
        print(f"   Description: {repo['description'] or 'No description'}")
        print(f"   Stars: {repo['stargazers_count']}")
        print(f"   URL: {repo['html_url']}")
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_repos.py <github_username>")
        sys.exit(1)
    
    username = sys.argv[1]
    print(f"Fetching repositories for user: {username}")
    repositories = fetch_user_repos(username)
    
    if not repositories:
        print("No repositories found.")
    else:
        print(f"Total repositories: {len(repositories)}")
        display_repos(repositories)