# Deployment Guide

## GitHub Repository Setup

1. Create a new repository on GitHub:
   - Go to https://github.com/new
   - Repository name: `tic-tac-toe`
   - Description: "Tic-Tac-Toe AI Lab - Interactive educational project exploring game-playing algorithms"
   - Visibility: Public
   - Do NOT initialize with README, .gitignore, or license (we already have these)

2. Push your code to GitHub:
```bash
git init
git add .
git commit -m "Initial commit: Tic-Tac-Toe AI Lab"
git branch -M main
git remote add origin https://github.com/rohanrgupte/tic-tac-toe.git
git push -u origin main
```

## Streamlit Cloud Deployment

1. Go to https://share.streamlit.io/
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `rohanrgupte/tic-tac-toe`
5. Main file path: `app/main.py`
6. App URL: The app will be available at `https://tic-tac-toe-ai-lab.streamlit.app` (or similar)
7. Click "Deploy"

### Streamlit Cloud Configuration

The app is configured via `.streamlit/config.toml` which is already included in the repository.

### Important Notes

- Make sure `requirements.txt` includes all dependencies
- The app will automatically redeploy when you push changes to the main branch
- Streamlit Cloud provides free hosting for public repositories

## Local Testing Before Deployment

Before deploying, test locally:

```bash
streamlit run app/main.py
```

Make sure everything works correctly, especially:
- All pages load without errors
- Agent selection works
- Simulations run correctly
- Q-learning training works
- MCTS visualization works

