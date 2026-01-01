import requests
import argparse
import sys

def fetch_repositories(username, sort_by='updated', order='desc'):
    url = f"https://api.github.com/users/{username}/repos"
    params = {'sort': sort_by, 'direction': order}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        repos = response.json()
        
        if not repos:
            print(f"No repositories found for user: {username}")
            return
            
        print(f"Repositories for {username} (sorted by {sort_by}, order: {order}):")
        print("-" * 60)
        for repo in repos:
            print(f"Name: {repo['name']}")
            print(f"  Description: {repo['description'] or 'No description'}")
            print(f"  Stars: {repo['stargazers_count']}")
            print(f"  Updated: {repo['updated_at']}")
            print(f"  URL: {repo['html_url']}")
            print()
            
    except requests.exceptions.HTTPError as e:
        print(f"Error fetching repositories: {e}")
        if response.status_code == 404:
            print(f"User '{username}' not found on GitHub")
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Fetch GitHub user repositories')
    parser.add_argument('username', help='GitHub username')
    parser.add_argument('--sort', choices=['created', 'updated', 'pushed', 'full_name'],
                       default='updated', help='Sort repositories by field')
    parser.add_argument('--order', choices=['asc', 'desc'], default='desc',
                       help='Sort order (ascending or descending)')
    
    args = parser.parse_args()
    
    fetch_repositories(args.username, args.sort, args.order)

if __name__ == "__main__":
    main()