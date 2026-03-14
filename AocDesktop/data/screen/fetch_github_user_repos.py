import requests
import sys

def fetch_repositories(username, page=1, per_page=30):
    url = f"https://api.github.com/users/{username}/repos"
    params = {
        'page': page,
        'per_page': per_page,
        'sort': 'updated',
        'direction': 'desc'
    }
    headers = {'Accept': 'application/vnd.github.v3+json'}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repositories: {e}", file=sys.stderr)
        return None

def display_repositories(repos):
    if not repos:
        print("No repositories found.")
        return
    
    for repo in repos:
        print(f"Name: {repo['name']}")
        print(f"  Description: {repo['description'] or 'No description'}")
        print(f"  Language: {repo['language'] or 'Not specified'}")
        print(f"  Stars: {repo['stargazers_count']}")
        print(f"  URL: {repo['html_url']}")
        print("-" * 50)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_user_repos.py <username> [page] [per_page]")
        sys.exit(1)
    
    username = sys.argv[1]
    page = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    per_page = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    
    repos = fetch_repositories(username, page, per_page)
    if repos is not None:
        display_repositories(repos)

if __name__ == "__main__":
    main()import requests

def fetch_user_repos(username, per_page=30, page=1):
    url = f"https://api.github.com/users/{username}/repos"
    params = {
        'per_page': per_page,
        'page': page
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()

def display_repos(repos):
    if not repos:
        print("No repositories found.")
        return
    for repo in repos:
        print(f"Name: {repo['name']}")
        print(f"Description: {repo['description'] or 'No description'}")
        print(f"URL: {repo['html_url']}")
        print(f"Stars: {repo['stargazers_count']}")
        print(f"Forks: {repo['forks_count']}")
        print("-" * 40)

def main():
    username = input("Enter GitHub username: ")
    try:
        page = 1
        while True:
            repos = fetch_user_repos(username, page=page)
            if not repos:
                print("No more repositories.")
                break
            print(f"\n--- Page {page} ---")
            display_repos(repos)
            if len(repos) < 30:
                break
            cont = input("Fetch next page? (y/n): ").strip().lower()
            if cont != 'y':
                break
            page += 1
    except requests.exceptions.HTTPError as e:
        print(f"Error fetching repositories: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()