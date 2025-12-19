import requests
import sys

def fetch_contributors(repo_owner, repo_name):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contributors"
    response = requests.get(url)
    
    if response.status_code == 200:
        contributors = response.json()
        print(f"Contributors for {repo_owner}/{repo_name}:")
        for contributor in contributors:
            print(f"- {contributor['login']}: {contributor['contributions']} contributions")
    else:
        print(f"Failed to fetch contributors. Status code: {response.status_code}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_github_contributors.py <repo_owner> <repo_name>")
        sys.exit(1)
    
    repo_owner = sys.argv[1]
    repo_name = sys.argv[2]
    fetch_contributors(repo_owner, repo_name)