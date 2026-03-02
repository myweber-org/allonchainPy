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
import requests
import sys

def fetch_github_repos(username, sort_by='created', direction='desc'):
    """
    Fetch public repositories for a given GitHub username.
    """
    url = f"https://api.github.com/users/{username}/repos"
    params = {
        'type': 'all',
        'sort': sort_by,
        'direction': direction,
        'per_page': 100
    }
    headers = {'Accept': 'application/vnd.github.v3+json'}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        repos = response.json()
        return repos
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
        return None
    except Exception as err:
        print(f"An error occurred: {err}")
        return None

def display_repos(repos):
    """
    Display repository information.
    """
    if not repos:
        print("No repositories to display.")
        return

    print(f"Found {len(repos)} repositories:")
    print("-" * 80)
    for repo in repos:
        name = repo.get('name', 'N/A')
        description = repo.get('description', 'No description')
        stars = repo.get('stargazers_count', 0)
        forks = repo.get('forks_count', 0)
        updated = repo.get('updated_at', 'N/A')[:10]
        print(f"Name: {name}")
        print(f"Description: {description}")
        print(f"Stars: {stars} | Forks: {forks} | Updated: {updated}")
        print("-" * 80)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_repos.py <github_username> [sort_by] [direction]")
        print("sort_by options: created, updated, pushed, full_name (default: created)")
        print("direction options: asc, desc (default: desc)")
        sys.exit(1)

    username = sys.argv[1]
    sort_by = sys.argv[2] if len(sys.argv) > 2 else 'created'
    direction = sys.argv[3] if len(sys.argv) > 3 else 'desc'

    repos = fetch_github_repos(username, sort_by, direction)
    if repos is not None:
        display_repos(repos)

if __name__ == "__main__":
    main()