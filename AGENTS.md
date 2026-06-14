# AGENTS - ICF GitHub Repository Guide

## Repository Information

- **GitHub URL**: https://github.com/icf/Share
- **Local Path**: `C:\_icf\icf_lib_AI_agent\Github`
- **Default Branch**: main (master 已合并)

---

## Git Workflow

### 1. Daily Sync Workflow

When files in `C:\_icf\icf_lib_AI_agent\Github` are updated, follow these steps to sync to GitHub:

```bash
# Step 1: Navigate to repository
cd C:\_icf\icf_lib_AI_agent\Github

# Step 2: Check status
git status

# Step 3: Stage all changes
git add .

# Step 4: Commit with descriptive message
git commit -m "Description of changes made"

# Step 5: Push to GitHub
git push origin main
```

### 2. Git Configuration

```bash
# User identity (already configured)
git config --global user.name "icf"
git config --global user.email "zxiaomain@outlook.com"

# Core settings
git config --global core.longpaths true
```

### 3. SSH Key

- SSH key location: `~/.ssh/id_ed25519`
- Public key: `~/.ssh/id_ed25519.pub`
- GitHub host key: `~/.ssh/known_hosts`

---

## Important Notes

1. **Before pushing**: Always run `git status` to see what files changed
2. **Commit messages**: Use clear, descriptive messages explaining the changes
3. **Large files**: If adding large files (>50MB), use Git LFS or reconsider
4. **Sensitive data**: Never commit secrets, API keys, or credentials
5. **New files**: Any new files added to this folder should be reviewed before committing

---

## Troubleshooting

### SSH Connection Issues
```bash
# Test SSH connection
ssh -T git@github.com

# Re-add GitHub host key if needed
ssh-keyscan github.com >> ~/.ssh/known_hosts
```

### Push Rejected (remote has changes)
```bash
# Option 1: Pull and merge first
git pull origin main
git merge origin/main
# Resolve conflicts if any, then push

# Option 2: Force push (use with caution)
git push -f origin main
```

### Check Current State
```bash
# See recent commits
git log --oneline -5

# See remote URLs
git remote -v

# See all branches
git branch -a
```
