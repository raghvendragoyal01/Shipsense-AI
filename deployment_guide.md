# 🚀 ShipSense AI — GitHub + Streamlit Cloud Deployment Guide

## What Gets Uploaded to GitHub

| File | Size | Needed? |
|------|------|---------|
| `app.py` | ~20 KB | ✅ Yes |
| `requirements.txt` | tiny | ✅ Yes |
| `assets/` (4 images) | ~38 KB | ✅ Yes |
| `data/classical_ml/best_rf.pkl` | 0.7 MB | ✅ Yes |
| `data/classical_ml/best_svm.pkl` | 0.6 MB | ✅ Yes |
| `data/classical_ml/best_knn.pkl` | 6.5 MB | ✅ Yes |
| `data/classical_ml/scaler.pkl` | tiny | ✅ Yes |
| `data/models/Custom_CNN.pth` | 0.5 MB | ✅ Yes |
| `data/models/ResNet18_Transfer.pth` | 44.8 MB | ✅ Yes |
| `data/tensors/y_labels.npy` | tiny | ✅ Yes |
| `data/tensors/class_mapping.npy` | tiny | ✅ Yes |
| `notebooks/Model_Testing_History.csv` | tiny | ✅ Yes |
| **`data/tensors/X_features.npy`** | **268 MB** | ❌ Excluded (.gitignore) |
| **`data/raw/`, `data/processed/`** | **GBs** | ❌ Excluded |
| **`pvr_env/`** | large | ❌ Excluded |

> [!IMPORTANT]
> Total upload ≈ **55 MB** — well within GitHub's 100MB file limit ✅

---

## STEP 1 — Install Git (if not already)

Open PowerShell and check:
```powershell
git --version
```
If not found → download from https://git-scm.com/download/win and install.

---

## STEP 2 — Create a GitHub Account & New Repo

1. Go to **https://github.com** → Sign up / Log in
2. Click the **"+"** button (top right) → **New repository**
3. Fill in:
   - **Repository name:** `shipsense-ai`  
   - **Description:** `Underwater acoustic vessel classification using ML and Deep Learning`
   - **Visibility:** Public *(required for free Streamlit Cloud)*
   - ❌ Do NOT check "Add README" or ".gitignore" — we already have them
4. Click **Create repository**
5. **Copy the repo URL** shown — looks like:  
   `https://github.com/YOUR_USERNAME/shipsense-ai.git`

---

## STEP 3 — Configure Git Identity (one-time setup)

```powershell
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

---

## STEP 4 — Initialize & Push from Project Folder

Run these commands **one by one** in PowerShell inside the project folder:

```powershell
# 1. Initialize git repo
git init

# 2. Add all files (respects .gitignore automatically)
git add .

# 3. Check what will be committed (verify X_features.npy is NOT listed)
git status

# 4. Commit
git commit -m "Initial commit: ShipSense AI with CNN + ResNet-18 + Classical ML"

# 5. Set branch name
git branch -M main

# 6. Link to your GitHub repo (replace with YOUR repo URL)
git remote add origin https://github.com/YOUR_USERNAME/shipsense-ai.git

# 7. Push
git push -u origin main
```

> [!NOTE]
> GitHub will ask for your username + password the first time.
> Use a **Personal Access Token** as the password (not your GitHub password).
> Get one at: GitHub → Settings → Developer Settings → Personal Access Tokens → Tokens (classic) → Generate new token → check `repo` scope.

---

## STEP 5 — Deploy on Streamlit Cloud

1. Go to **https://share.streamlit.io** → Sign in with GitHub
2. Click **"New app"**
3. Fill in:
   - **Repository:** `YOUR_USERNAME/shipsense-ai`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy!**
5. Wait ~3–5 minutes for dependencies to install (torch is large)
6. Your app will be live at:  
   `https://YOUR_USERNAME-shipsense-ai-app-XXXX.streamlit.app`

---

## STEP 6 — For Future Updates

After changing `app.py` or any file locally:

```powershell
git add .
git commit -m "Update: describe your change here"
git push
```
Streamlit Cloud **auto-redeploys** on every push. ✅

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `git push` asks for password | Use Personal Access Token, not GitHub password |
| File too large error | Check `.gitignore` has `X_features.npy` — run `git status` to verify |
| Streamlit Cloud build fails | Check `requirements.txt` has all packages |
| `ModuleNotFoundError` on cloud | Add missing package to `requirements.txt` and push again |
| App slow on first load | Normal — PyTorch models load once then are cached |

> [!TIP]
> After pushing, you can share the Streamlit URL with anyone — no installation needed on their side!

---

## 💾 Handling Large Dataset Folders (Optional)

If you need to upload or share your entire dataset folders (`data/raw/` and `data/processed/`), which are currently blocked by `.gitignore` due to their massive size (GBs), you have a few options:

### Option 1: Cloud Storage (Recommended)
GitHub has a hard limit of 100MB per file and a general repository limit. Instead of uploading datasets to GitHub:
1. Upload the `data/` folder directly to **Google Drive**, **OneDrive**, **AWS S3**, or **Kaggle Datasets**.
2. Share the download link with your team or include it in your project's `README.md`.
3. If your Streamlit app needs the raw data (usually it doesn't, it only needs the trained models), you can use Python libraries like `gdown` or `boto3` to dynamically download the data when the app starts.

### Option 2: Git LFS (Large File Storage)
If you *must* track datasets via Git, you can use Git LFS. **Warning:** GitHub only provides 1GB of free LFS storage and 1GB/month of bandwidth.
```powershell
# 1. Install Git LFS
git lfs install

# 2. Track the large files
git lfs track "data/raw/**"
git lfs track "data/processed/**"

# 3. Add and commit
git add .gitattributes data/raw data/processed
git commit -m "Add large datasets via LFS"
git push origin main
```

### Option 3: Hugging Face Spaces / Datasets
If your dataset is larger than GitHub's limits, consider moving the deployment to **Hugging Face Spaces**. Hugging Face is designed for ML and provides native dataset hosting without strict file size limits.
