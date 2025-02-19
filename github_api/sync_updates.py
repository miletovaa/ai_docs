from github_api.check_updates import check_for_new_commits

def sync_updates():
    print("🔍 Checking for new commits...")
    has_new_commits = check_for_new_commits()

    if has_new_commits:
        print("🚀 New commits detected! Proceeding with sync...")
    else:
        print("✅ No new commits found. Everything is up to date.")

if __name__ == "__main__":
    sync_updates()