import requests
import sys

def fetch_repo_info(owner, repo):
    """
    Fetch basic information about a GitHub repository.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        info = {
            "name": data.get("name"),
            "full_name": data.get("full_name"),
            "description": data.get("description"),
            "html_url": data.get("html_url"),
            "stargazers_count": data.get("stargazers_count"),
            "forks_count": data.get("forks_count"),
            "open_issues_count": data.get("open_issues_count"),
            "language": data.get("language"),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at")
        }
        return info
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except Exception as err:
        print(f"An error occurred: {err}")
        return None

def main():
    if len(sys.argv) != 3:
        print("Usage: python fetch_github_repo_info.py <owner> <repo>")
        sys.exit(1)

    owner = sys.argv[1]
    repo = sys.argv[2]

    repo_info = fetch_repo_info(owner, repo)

    if repo_info:
        print(f"Repository: {repo_info['full_name']}")
        print(f"Description: {repo_info['description']}")
        print(f"URL: {repo_info['html_url']}")
        print(f"Stars: {repo_info['stargazers_count']}")
        print(f"Forks: {repo_info['forks_count']}")
        print(f"Open Issues: {repo_info['open_issues_count']}")
        print(f"Language: {repo_info['language']}")
        print(f"Created: {repo_info['created_at']}")
        print(f"Last Updated: {repo_info['updated_at']}")
    else:
        print("Failed to fetch repository information.")

if __name__ == "__main__":
    main()