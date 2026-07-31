# 6th place solution - Ultralytics YOLO 

Thanks to kaggle and everyone. I start by host's sample code, and train several Ultralytics' YOLO models by different configs.

# Summary
- Ultralytics YOLO models
- Apply filter for denoise
- More data augmentation
- Thresholding strategy
- External data by @brendanartley 

# Data process
- Use [z-3, z, z+3] slices as RGB channels input.
- Use hamming window to filter data along the z-axis to perform denoise.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F933653%2F8777d7dbedc2e89fb9abcfe106c6fcbe%2F2025-07-01%20092426.jpg?generation=1751333082738327&alt=media)

# Augmentation
Modify ultralytics/data/base.py for more augmentation, like gamma and random size.

# Thresholding
Two strategy, both perform well:
- Thresholding by 56 percentile
- Auto Threshold：
`
s = np.sort(confidence_score)
`
`
r = [ ( s[i-50] + s[i+50] - 2*s[i] ) for i in range(50, len(s)-200) ]
`
`
confidence_threshold = s[ np.argmax(r) + 50 ]
`

# Ensemble
The private LB score is ensemble by 4 models:
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F933653%2F8c7f0c2d11c8eccd6a78c2b7b6afe5f0%2F2025-07-01%20092617.jpg?generation=1751333257096816&alt=media)

Thanks again.