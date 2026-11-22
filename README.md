On the 2-class Brain MRI dataset, MQAdapter did not perform well. When trained on the full training set, it lowered the accuracy of all three methods it was added to, and the best result came from the simplest baseline.

Reason : with only 2 classes, MQAdapter’s Top-K step always keeps both classes and has no effect, and that the test set is very small (45 images) so the results vary a lot between runs.



Research paper using it: https://www.nature.com/articles/s41598-026-60334-y#Sec4
dataset  link: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/data?select=yes