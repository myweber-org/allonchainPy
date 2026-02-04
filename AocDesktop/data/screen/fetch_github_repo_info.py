import requests
import sys

def fetch_repo_info(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        print(f"Repository: {data['full_name']}")
        print(f"Description: {data['description']}")
        print(f"Stars: {data['stargazers_count']}")
        print(f"Forks: {data['forks_count']}")
        print(f"Open Issues: {data['open_issues_count']}")
        print(f"Language: {data['language']}")
        print(f"URL: {data['html_url']}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repository info: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_github_repo_info.py <owner> <repo>", file=sys.stderr)
        sys.exit(1)
    
    owner = sys.argv[1]
    repo = sys.argv[2]
    fetch_repo_info(owner, repo)