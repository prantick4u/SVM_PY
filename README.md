# SVM_PY
Some rudimentary work using SVM classifier 
Here we are having a hands on exploration on SVM using PY libs and understanding few key points on the same.

We built this **Support Vector Machine** for **classification** using **scikit-learn** and the **Radial Basis Function (RBF) Kernel**. 
Our training data set contains continuous and categorical data from the **[UCI Machine Learning 
Repository](https://archive.ics.uci.edu/ml/index.php)** to predict whether or not a patient has **[heart disease](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)**.

Steps:
- Importing the Data From a File
- Identifying Missing Data
- Dealing with Missing Data
- Split data into Dependent and Independent Variables 
- One-Hot-Encoding
- Centering and Scaling the Data
- Building a Preliminary Support Vector Machine
- Opimizing Parameters with Cross Validation (Cross Validation For Finding the Best Values for Gamma and Regularization)
- Building, Evaluating, Drawing and Interpreting the Final Support Vector Machine


N.B. We need to install the following dependencies:
  - python=3.6
  - pandas
  - numpy
  - matplotlib
  - scikit-learn

Results:

Predicted Data vs Actual Data:

![Figure_2](https://user-images.githubusercontent.com/18325530/118940629-898a1080-b96e-11eb-8a6e-81f036319ebc.png)


Graphical representation of percentage of explained variance vs degree of components:

![Figure_1](https://user-images.githubusercontent.com/18325530/118940635-8abb3d80-b96e-11eb-92b9-244290e1794e.png)
