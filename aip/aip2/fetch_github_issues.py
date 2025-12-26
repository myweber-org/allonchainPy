import requests
import sys

def fetch_issues(owner, repo, count=5):
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    params = {'state': 'open', 'per_page': count}
    headers = {'Accept': 'application/vnd.github.v3+json'}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        issues = response.json()
        return issues
    except requests.exceptions.RequestException as e:
        print(f"Error fetching issues: {e}", file=sys.stderr)
        return []

def display_issues(issues):
    if not issues:
        print("No issues found.")
        return

    for issue in issues:
        print(f"#{issue['number']}: {issue['title']}")
        print(f"    State: {issue['state']}")
        print(f"    Created: {issue['created_at']}")
        print(f"    URL: {issue['html_url']}")
        print()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_github_issues.py <owner> <repo>")
        sys.exit(1)

    owner = sys.argv[1]
    repo = sys.argv[2]
    issues = fetch_issues(owner, repo)

    if issues:
        print(f"Recent open issues for {owner}/{repo}:\n")
        display_issues(issues)