import requests
import subprocess
import os

# GitHub 저장소 정보
OWNER = "user"
REPO = "repo"
BRANCH = "main"  # 추적할 브랜치
TOKEN = "token"  # 개인 액세스 토큰 (필요시)

# GitHub API URL
COMMITS_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/commits?sha={BRANCH}"
# print(COMMITS_URL)
# 로컬 커밋 해시 저장 파일
LOCAL_COMMIT_FILE = "last_commit.txt"

def get_latest_commit():
    headers = {"Authorization": f"token {TOKEN}"} if TOKEN else {}
    response = requests.get(COMMITS_URL, headers=headers)
    response.raise_for_status()
    data = response.json()
    return data[0]["sha"]  # 최신 커밋의 SHA 반환

def read_local_commit():
    if os.path.exists(LOCAL_COMMIT_FILE):
        with open(LOCAL_COMMIT_FILE, "r") as file:
            return file.read().strip()
    return None

def write_local_commit(commit_sha):
    with open(LOCAL_COMMIT_FILE, "w") as file:
        file.write(commit_sha)

def update_code():
    result = subprocess.run(["git", "pull"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Error updating the repository:", result.stderr)

def main():
    latest_commit = get_latest_commit()
    local_commit = read_local_commit()

    if local_commit != latest_commit:
        print("New update detected. Updating the code...")
        update_code()
        write_local_commit(latest_commit)
    else:
        print("Code is already up-to-date.")

if __name__ == "__main__":
    main()