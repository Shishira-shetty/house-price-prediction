import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import random
import os

# Step 1: Load and preprocess the dataset
data = pd.read_csv('Bengaluru_House_Data.csv')
data = data.drop(['area_type', 'availability', 'society', 'balcony'], axis=1)
data = data.dropna()

# Clean 'size' column
data['bhk'] = data['size'].apply(lambda x: int(x.split(' ')[0]))
data = data.drop('size', axis=1)

# Clean 'total_sqft'
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None

data['total_sqft'] = data['total_sqft'].apply(convert_sqft_to_num)
data = data.dropna()

# Simplify 'location' and get top 10 locations
top_locations = data['location'].value_counts().index[:10]
data['location'] = data['location'].apply(lambda x: x.strip() if x in top_locations else 'Other')

# One-hot encode 'location'
data = pd.get_dummies(data, columns=['location'], drop_first=True)

# Step 2: Define features (X) and target (y)
X = data.drop('price', axis=1)
y = data['price']

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test_scaled)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Step 8: Prediction function
def predict_price(location, total_sqft, bath, bhk):
    input_data = pd.DataFrame(columns=X.columns)
    input_data.loc[0] = 0
    
    input_data['total_sqft'] = total_sqft
    input_data['bath'] = bath
    input_data['bhk'] = bhk
    
    if location != 'Other' and f'location_{location}' in X.columns:
        input_data[f'location_{location}'] = 1
    
    input_data = input_data.fillna(0)
    input_data_scaled = scaler.transform(input_data)
    predicted_price = model.predict(input_data_scaled)[0]
    return predicted_price

# Step 9: Tkinter GUI with Background
def predict_price_gui():
    try:
        total_sqft = float(entry_sqft.get())
        bath = float(entry_bath.get())
        bhk = float(entry_bhk.get())
        location = combo_location.get()
        
        predicted_price = predict_price(location, total_sqft, bath, bhk)
        result_label.config(text=f"Predicted Price: ₹{predicted_price:.2f} Lakhs")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for Sqft, Bath, and BHK")

# Function to change background image
def change_background():
    global bg_images, bg_label
    image_path = random.choice(bg_images)
    img = Image.open(image_path)
    img = img.resize((1400, 1300), Image.Resampling.LANCZOS)  # Resize to fit window
    photo = ImageTk.PhotoImage(img)
    bg_label.config(image=photo)
    bg_label.image = photo  # Keep a reference to avoid garbage collection
    root.after(4000, change_background)  # Change every 4 seconds (4000 ms)

# Create the main window
root = tk.Tk()
root.title("Bengaluru House Price Predictor")
root.geometry("400x300")

# Load background images from a folder (create a 'backgrounds' folder with images)
bg_folder = "backgrounds"  # Folder name
if not os.path.exists(bg_folder):
    os.makedirs(bg_folder)
    print(f"Created '{bg_folder}' folder. Please add some background images (e.g., .jpg, .png) to it.")
    bg_images = []  # No images yet
else:
    bg_images = [os.path.join(bg_folder, f) for f in os.listdir(bg_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Add background label
bg_label = tk.Label(root)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)  # Cover entire window

if bg_images:
    change_background()  # Start changing background if images are available
else:
    print("No background images found. Running with plain background.")

# Labels and Entry fields (place over background)
tk.Label(root, text="Total Sqft:", bg="white").pack(pady=5)
entry_sqft = tk.Entry(root)
entry_sqft.pack()

tk.Label(root, text="Number of Bathrooms:", bg="white").pack(pady=5)
entry_bath = tk.Entry(root)
entry_bath.pack()

tk.Label(root, text="Number of BHK:", bg="white").pack(pady=5)
entry_bhk = tk.Entry(root)
entry_bhk.pack()

tk.Label(root, text="Location:", bg="white").pack(pady=5)
combo_location = ttk.Combobox(root, values=list(top_locations) + ['Other'])
combo_location.set("Other")
combo_location.pack()

# Predict Button
predict_button = tk.Button(root, text="Predict Price", command=predict_price_gui)
predict_button.pack(pady=20)

# Result Label
result_label = tk.Label(root, text="Predicted Price: ₹0.00 Lakhs", font=("Arial", 12, "bold"), bg="white")
result_label.pack(pady=10)

# Run the GUI
root.mainloop()

# Step 10: Visualize Actual vs Predicted Prices (optional, runs before GUI)
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price (Lakhs)')
plt.ylabel('Predicted Price (Lakhs)')
plt.title('Actual vs Predicted House Prices in Bengaluru')
plt.tight_layout()
plt.show()
