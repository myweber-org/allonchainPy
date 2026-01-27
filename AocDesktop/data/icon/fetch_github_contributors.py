import requests
import sys

def get_top_contributors(repo_owner, repo_name, top_n=5):
    """
    Fetch top contributors for a GitHub repository based on commit count.
    """
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contributors"
    headers = {"Accept": "application/vnd.github.v3+json"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        contributors = response.json()
        
        if not contributors:
            print(f"No contributors found for {repo_owner}/{repo_name}")
            return []
        
        sorted_contributors = sorted(
            contributors,
            key=lambda x: x.get("contributions", 0),
            reverse=True
        )[:top_n]
        
        return sorted_contributors
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return []
    except ValueError as e:
        print(f"Error parsing response: {e}")
        return []

def display_contributors(contributors, repo_owner, repo_name):
    """
    Display contributor information in a formatted way.
    """
    if not contributors:
        return
    
    print(f"Top {len(contributors)} contributors for {repo_owner}/{repo_name}:")
    print("-" * 50)
    
    for i, contributor in enumerate(contributors, 1):
        username = contributor.get("login", "Unknown")
        contributions = contributor.get("contributions", 0)
        profile_url = contributor.get("html_url", "")
        
        print(f"{i}. {username}")
        print(f"   Contributions: {contributions}")
        print(f"   Profile: {profile_url}")
        print()

def main():
    if len(sys.argv) != 3:
        print("Usage: python fetch_github_contributors.py <repo_owner> <repo_name>")
        print("Example: python fetch_github_contributors.py torvalds linux")
        sys.exit(1)
    
    repo_owner = sys.argv[1]
    repo_name = sys.argv[2]
    
    contributors = get_top_contributors(repo_owner, repo_name)
    display_contributors(contributors, repo_owner, repo_name)

if __name__ == "__main__":
    main()