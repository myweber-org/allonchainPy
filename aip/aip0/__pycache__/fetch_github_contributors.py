import requests
import sys

def get_top_contributors(repo_owner, repo_name, top_n=5):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contributors"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        contributors = response.json()
        
        if not contributors:
            print("No contributors found.")
            return []
        
        sorted_contributors = sorted(
            contributors, 
            key=lambda x: x.get('contributions', 0), 
            reverse=True
        )[:top_n]
        
        return sorted_contributors
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return []
    except ValueError as e:
        print(f"Error parsing response: {e}")
        return []

def display_contributors(contributors):
    if not contributors:
        return
    
    print(f"{'Rank':<5} {'Username':<20} {'Contributions':<15}")
    print("-" * 45)
    
    for idx, contributor in enumerate(contributors, 1):
        username = contributor.get('login', 'N/A')
        contributions = contributor.get('contributions', 0)
        print(f"{idx:<5} {username:<20} {contributions:<15}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fetch_github_contributors.py <repo_owner> <repo_name> [top_n]")
        sys.exit(1)
    
    repo_owner = sys.argv[1]
    repo_name = sys.argv[2]
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    contributors = get_top_contributors(repo_owner, repo_name, top_n)
    display_contributors(contributors)