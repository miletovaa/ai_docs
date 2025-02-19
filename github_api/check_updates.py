import os
import requests
import mysql.connector
import json
from dotenv import load_dotenv
from db.connector import get_db_connection

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_OWNER = os.getenv("GITHUB_OWNER")
GITHUB_REPO = os.getenv("GITHUB_REPO")

GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/commits"

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def get_last_commit_from_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT commit_hash FROM github_history LIMIT 1;")
        result = cursor.fetchone()

        cursor.close()
        conn.close()

        return result[0] if result else None

    except mysql.connector.Error as err:
        print(f"❌ Database Error: {err}")
        return None

def get_new_commits_from_github(since_commit):
    try:
        new_commits = []
        page = 1

        while True:
            url = f"{GITHUB_API_URL}?per_page=100&page={page}"
            response = requests.get(url, headers=HEADERS)

            if response.status_code == 200:
                commits = response.json()
                
                for commit in commits:
                    sha = commit["sha"]
                    if sha == since_commit:
                        return new_commits
                    new_commits.append(sha)

                if len(commits) < 100:
                    break
                page += 1
            else:
                print(f"❌ GitHub API Error: {response.status_code} - {response.json()}")
                return None

        return new_commits

    except Exception as e:
        print(f"❌ Request Error: {e}")
        return None

def check_for_new_commits():
    last_commit = get_last_commit_from_db()

    if not last_commit:
        print("⚠️ No commit hash found in the database.")
        return False

    latest_commits = get_new_commits_from_github(last_commit)

    if not latest_commits:
        print("✅ No new commits found.")
        return False

    print(f"🚀 New commits found ({len(latest_commits)}):")
    print(json.dumps(latest_commits, indent=4))
    return True

if __name__ == "__main__":
    check_for_new_commits()