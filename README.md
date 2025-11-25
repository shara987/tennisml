# tennisml

PUSH WITH LFS

🚀 Git LFS Workflow (Windows)

```bash
git init
git lfs install
git lfs track "*.mp4"
git lfs track "*.onnx"
git lfs track "*.zip"
git lfs track "<file_name>"
git rm --cached largefile.mp4
git lfs migrate import --include="frames.zip" --include-ref=refs/heads/main
git lfs ls-files
git ls-tree -r HEAD | grep <file_name>
git add .gitattributes
git commit -m "Configure Git LFS and migrate large files"
git push origin <branch> --force
# or
git push origin main

# Git LFS Workflow (Linux)

```bash
git init
git lfs install
git lfs track "*.mp4"
git lfs track "*.onnx"
git lfs track "*.zip"
git lfs track "<file_name>"
git rm --cached largefile.mp4
git lfs migrate import --include="frames.zip" --include-ref=refs/heads/main
git lfs ls-files
git ls-tree -r HEAD | grep <file_name>
git add .gitattributes
git commit -m "Configure Git LFS and migrate large files"
git push origin <branch> --force
# or
git push origin main

# Git LFS Workflow (macOS)

```bash
git init
git lfs install
git lfs track "*.mp4"
git lfs track "*.onnx"
git lfs track "*.zip"
git lfs track "<file_name>"
git rm --cached largefile.mp4
git lfs migrate import --include="frames.zip" --include-ref=refs/heads/main
git lfs ls-files
git ls-tree -r HEAD | grep <file_name>
git add .gitattributes
git commit -m "Configure Git LFS and migrate large files"
git push origin <branch> --force
# or
git push origin main

PULL WITH LFS

