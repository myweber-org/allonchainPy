import requests
import sys
from datetime import datetime, timedelta

def fetch_recent_issues(owner, repo, days=7):
    """
    Fetch open issues created in the last N days from a GitHub repository.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    since_date = (datetime.now() - timedelta(days=days)).isoformat()
    params = {
        'state': 'open',
        'since': since_date,
        'per_page': 50
    }
    headers = {'Accept': 'application/vnd.github.v3+json'}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        issues = response.json()
        return [issue for issue in issues if 'pull_request' not in issue]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching issues: {e}", file=sys.stderr)
        return []

def display_issues(issues):
    """
    Display issue details in a simple formatted way.
    """
    if not issues:
        print("No recent open issues found.")
        return

    print(f"Found {len(issues)} recent open issue(s):\n")
    for idx, issue in enumerate(issues, 1):
        print(f"{idx}. #{issue['number']}: {issue['title']}")
        print(f"   Created: {issue['created_at']}")
        print(f"   URL: {issue['html_url']}")
        print(f"   Author: {issue['user']['login']}")
        print()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_github_issues.py <owner> <repo>")
        sys.exit(1)

    owner = sys.argv[1]
    repo = sys.argv[2]
    issues = fetch_recent_issues(owner, repo)
    display_issues(issues)