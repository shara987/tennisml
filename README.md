# tennisml

PUSH WITH LFS

Window:
git init
git lfs install
git lfs track "<file wanna upload>" -> for this use *.mp4 or *.onnx (if we planned to use onnx to upload the model)
or even git lfs track "*.extension" if too big?
git rm --cached largefile.mp4
git add .gitattributes or git add .
git commit -m "<a command here>" -> based on the person who push
git push or git push origin main



PULL WITH LFS

