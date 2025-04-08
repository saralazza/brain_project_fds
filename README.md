# 🧠✨ Synthetic Brain MRI Generation & Classification with AI

## 📄 Abstract

This project leverages **Artificial Intelligence** to address ethical and privacy concerns in healthcare by generating **synthetic brain cancer images**. The aim is to create realistic synthetic datasets that **mimic real patient data**, enabling effective training of ML models **without compromising privacy**.

---

## 🧠 Introduction

Training machine learning models on sensitive healthcare data raises serious **ethical and legal** issues.  
To tackle this, we explored the use of **Variational Autoencoders (VAE)** and **Generative Adversarial Networks (GAN)** to generate synthetic **brain MRI scans**.  
We then evaluated the performance of **classification models**—notably **Convolutional Neural Networks (CNN)** and VAEs—on **real, synthetic, and hybrid datasets**.

---

## 🗂️ Dataset

We used the **Br35H dataset**, which includes:

- 🧪 3000 MRI scans
    
    - 🧠 1500 tumor-positive
        
    - ✅ 1500 tumor-negative
        
- 📏 Varied image sizes & scanning techniques
    
- ⚠️ Non-IID structure, posing challenges for consistent model training
    

### 🖼️ Example Images

- **(a)** 'Flair' Brain MRI — _Size: 587×630_
    
- **(b)** 'T2' Brain MRI — _Size: 197×256_
    

---

## 🧪 Methodology

### 🧼 Pre-processing

- Resized and normalized all images to `[0, 1]`
    
- Maintained aspect ratios
    
- Unified backgrounds and dimensions
    

### 🧬 Data Generation

- **GAN (Generative Adversarial Network)**
    
    - Generator vs Discriminator in a min-max game
        
- **VAE (Variational Autoencoder)**
    
    - Encoder-decoder architecture mapping to latent space
        

🔁 Trained **separately for tumor-positive and tumor-negative** images to ensure balance

### 🧠 Classification

#### 🔍 Models

- **CNN Architecture**: Feature extraction + classification layers
    
- **VAE for Classification**: Used reconstruction error as a signal for anomaly (tumor) detection
    

#### 🛠️ Techniques

- 📏 **Global Image Absolute Error Magnitude (GIAEM)**
    
- 🌐 **DBSCAN**: Density-based spatial clustering
    
- 🎯 **Singular-example KMeans**
    
- 📊 **Global KMeans**: Clustered reconstruction errors across dataset
    

---

## 📊 Experimental Results

### 🎨 Generative Task

|Model|Result|
|---|---|
|**GAN**|❌ Struggled with realistic tumor generation, high noise|
|**VAE**|✅ More realistic images, but difficulties in tumor regions|

### 🤖 Classification Task

|Model|Performance|
|---|---|
|**CNN**|🟢 High accuracy on real data, poor generalization to synthetic|
|**VAE**|🟡 Varying results; Global KMeans achieved best balance|

---

### 📈 Performance Metrics

|Model|Train Data|Accuracy (%)|Precision (%)|Recall (%)|F1-score (%)|
|---|---|---|---|---|---|
|**CNN**|Real|**96.67**|95.42|97.99|96.69|
||Synthetic|58.16|57.53|56.94|57.24|
||Mixed|49.97|49.98|**99.93**|66.64|
|**VAE (GIAEM)**|Real|60.44|60.95|67.11|56.76|
||Synthetic|80.08|68.26|66.22|67.09|
||Mixed|**81.24**|**69.99**|64.11|65.95|
|**VAE (Global KMeans)**|Real|**82.93**|**73.55**|67.58|69.66|
||Synthetic|79.39|68.26|69.44|68.80|
||Mixed|78.80|39.88|49.27|44.08|

---

## 🧾 Conclusions & Future Work

- ⚠️ The **Br35H dataset** posed difficulties due to its heterogeneous nature
    
- 🔍 **VAE** proved to be the more effective generative model
    
- 🧠 **CNN** excelled on real data but lacked generalization
    
- 🔬 **Future work** should:
    
    - Explore **more consistent datasets**
        
    - Enhance **GAN performance**
        
    - Expand **unsupervised techniques** like DBSCAN and KMeans
        

---

## 📚 References

- _CovidGAN: Data Augmentation Using Auxiliary Classifier GAN for Improved Covid-19 Detection_ – Abdul Waheed et al.
    
- _Synthetic Medical Images Using F&BGAN for Improved Lung Nodules Classification_ – Defang Zhao et al.
    
- _Combining Noise-to-Image and Image-to-Image GANs: Brain MR Image Augmentation for Tumor Detection_
    
- _Infinite Brain MR Images: PGGAN-based Data Augmentation for Tumor Detection_
    
- _Br35H: Brain Tumor Detection 2020_
    
- DBSCAN
    
- KMeans
    

---

## 👨‍💻 Team

- 🧬 **Generation Team**: Francesco D'Aprile, Sara Lazzaroni
    
- 🔎 **Classification Team**: Anthony Di Pietro, Tommaso Mattei
    

---

## Sources

At the following Google Drive link, you can find the dataset (both real and synthetic) and the weights for the classification and generation models. https://drive.google.com/drive/folders/1JJT4MP_5GSH_CU1Blvm6GpfhzZvHZFlb

## 📎 More Info

🔗 _Check out the full paper in doc/ for more details!_
