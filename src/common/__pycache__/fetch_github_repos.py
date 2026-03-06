import requests
import sys

def fetch_repositories(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    if response.status_code == 200:
        repos = response.json()
        for repo in repos:
            print(f"Name: {repo['name']}")
            print(f"Description: {repo['description']}")
            print(f"URL: {repo['html_url']}")
            print(f"Stars: {repo['stargazers_count']}")
            print("-" * 40)
    else:
        print(f"Failed to fetch repositories. Status code: {response.status_code}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fetch_github_repos.py <username>")
        sys.exit(1)
    username = sys.argv[1]
    fetch_repositories(username)
import requests
import argparse
from datetime import datetime

def fetch_user_repositories(username, sort_by='created', direction='desc'):
    """
    Fetch public repositories for a given GitHub username.
    Allows sorting by 'created', 'updated', 'pushed', or 'full_name'.
    Direction can be 'asc' or 'desc'.
    """
    base_url = f"https://api.github.com/users/{username}/repos"
    params = {
        'type': 'public',
        'sort': sort_by,
        'direction': direction,
        'per_page': 100
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        repos = response.json()
        
        if not repos:
            print(f"No public repositories found for user '{username}'.")
            return []
            
        return repos
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repositories: {e}")
        return []

def display_repositories(repos, show_details=False):
    """
    Display repository information in a formatted way.
    """
    if not repos:
        return
    
    print(f"\nFound {len(repos)} repositories:")
    print("-" * 80)
    
    for i, repo in enumerate(repos, 1):
        print(f"{i}. {repo['name']}")
        print(f"   Description: {repo['description'] or 'No description'}")
        print(f"   Language: {repo['language'] or 'Not specified'}")
        print(f"   Stars: {repo['stargazers_count']} | Forks: {repo['forks_count']}")
        print(f"   Created: {datetime.strptime(repo['created_at'], '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d')}")
        print(f"   Updated: {datetime.strptime(repo['updated_at'], '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d')}")
        
        if show_details:
            print(f"   URL: {repo['html_url']}")
            print(f"   Default Branch: {repo['default_branch']}")
            print(f"   Size: {repo['size']} KB")
            print(f"   Open Issues: {repo['open_issues_count']}")
            print(f"   Archived: {repo['archived']}")
            print(f"   Fork: {repo['fork']}")
        
        print()

def main():
    parser = argparse.ArgumentParser(description='Fetch GitHub user repositories')
    parser.add_argument('username', help='GitHub username')
    parser.add_argument('--sort', choices=['created', 'updated', 'pushed', 'full_name'],
                       default='created', help='Sort repositories by field')
    parser.add_argument('--direction', choices=['asc', 'desc'],
                       default='desc', help='Sort direction')
    parser.add_argument('--details', action='store_true',
                       help='Show detailed repository information')
    parser.add_argument('--limit', type=int, default=0,
                       help='Limit number of repositories displayed (0 for all)')
    
    args = parser.parse_args()
    
    repos = fetch_user_repositories(args.username, args.sort, args.direction)
    
    if repos:
        if args.limit > 0:
            repos = repos[:args.limit]
        
        display_repositories(repos, args.details)
        
        # Print summary statistics
        total_stars = sum(repo['stargazers_count'] for repo in repos)
        total_forks = sum(repo['forks_count'] for repo in repos)
        languages = {}
        
        for repo in repos:
            lang = repo['language'] or 'Unknown'
            languages[lang] = languages.get(lang, 0) + 1
        
        print("\n" + "=" * 80)
        print(f"SUMMARY for '{args.username}':")
        print(f"  Total Repositories: {len(repos)}")
        print(f"  Total Stars: {total_stars}")
        print(f"  Total Forks: {total_forks}")
        print(f"  Languages Used: {', '.join(languages.keys())}")

if __name__ == "__main__":
    main()