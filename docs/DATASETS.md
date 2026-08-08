# Datasets

Recommended starting set:

1. **FaceForensics++** — main experimental dataset  
2. **Celeb-DF** — realistic high-quality deepfakes  

Optional:

3. **DFDC** — large-scale  
4. **ForgeryNet** — broader forgery research  

## Expected folder layout for training scripts

```
data/
  train/real/
  train/fake/
  val/real/
  val/fake/
  test/real/
  test/fake/
```

Preprocess with face detection → crop → resize 224 → normalize before training.
