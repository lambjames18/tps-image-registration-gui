import subprocess

def get_git_hash(short=False):
    cmd = ["git", "rev-parse", "HEAD"]
    if short:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
    return subprocess.check_output(cmd).decode("utf-8").strip()

print(get_git_hash())        # full hash
print(get_git_hash(short=True))  # short hash