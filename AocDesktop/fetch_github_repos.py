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
    headers = {
        'Accept': 'application/vnd.github.v3+json'
    }
    
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
        name = repo.get('name', 'N/A')
        description = repo.get('description', 'No description')
        stars = repo.get('stargazers_count', 0)
        forks = repo.get('forks_count', 0)
        language = repo.get('language', 'Not specified')
        updated = repo.get('updated_at', '')[:10]
        
        print(f"Repository: {name}")
        print(f"  Description: {description}")
        print(f"  Stars: {stars} | Forks: {forks} | Language: {language}")
        print(f"  Last updated: {updated}")
        print("-" * 50)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_github_repos.py <github_username> [page_number]")
        sys.exit(1)
    
    username = sys.argv[1]
    page = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    print(f"Fetching repositories for user '{username}' (Page {page})...")
    repos = fetch_repositories(username, page)
    
    if repos is not None:
        display_repositories(repos)
        
        if len(repos) == 30:
            print(f"\nNote: Showing page {page}. There may be more repositories.")
            print(f"To see next page, run: python {sys.argv[0]} {username} {page + 1}")

if __name__ == "__main__":
    main()