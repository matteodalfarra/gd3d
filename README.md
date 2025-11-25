# GD3D

A simple Python project to visualize **gradient descent** on a 3D cost surface for linear regression.

![GD3D](assets/gd3d.png)

---

## Description

This project demonstrates how gradient descent minimizes a cost function.  
It plots:

- A **3D surface** of the cost function \(J(w, b)\)
- The **gradient descent trajectory** in red on top of the surface

The visualization helps to understand how gradient descent converges to the minimum.

---

## Features

- Gradient descent implementation in pure Python
- 3D surface plot using `matplotlib`
- Adjustable learning rate and number of iterations
- Clear visualization of the optimization path

---

## Installation

Make sure you have Python 3.13.9 installed. Install the requirements if not already installed:

```bash
pip install -r requirements.txt
```

Clone the repository:
```
git clone https://github.com/matteodalfarra/gd3d.git
cd gd3d
```
## Usage
Run the main script:
```bash
python main.py
```

You will see a 3D plot showing the cost surface and the path of gradient descent.

---

## Parameters
- w: started weight 
- b: started bias 
- a: learning rate 
- n: number of gradient descent iterations
- points: data points used for linear regression
