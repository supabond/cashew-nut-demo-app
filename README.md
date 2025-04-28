# Cashew Nut Quality Assessment Web App

![image](https://github.com/user-attachments/assets/9f3f4ab5-b40b-42b7-a892-d3cb24dd3963)

This web application assesses the quality of cashew nuts using AI. It takes an uploaded image of cashew nuts, performs predictions, and displays the results, including relevant information fetched from a Google Sheet.

## Steps to Run the Application

### 1. Clone the Project to Your Local Machine
Clone the repository using the following command:

```bash
git clone https://github.com/your-username/cashew-nut-demo-app.git
```
### 2. Navigate to the Project Directory
Open the command prompt/terminal and navigate to the project directory:

```bash
cd cashew-nut-demo-app
```
### 3. Install Streamlit (If Not Installed)
If you do not have Streamlit installed on your system, you can install it using pip:

```bash
pip install streamlit
```

### 4. Run the Application
Once Streamlit is installed, you can run the application using the following command:

```bash
streamlit run app.py
```

This will launch the app in a web browser, and a pop-up window should appear with the web application ( please use light mode in browser ).

### 5. Upload Cashew Nut Image
To use the app, upload a cashew nut image either through the file upload box provided in the app or directly into the data folder in the project directory. Make sure the image title follows the naming conventions used in the Google Drive to retrieve accurate data.

### 6. Data in Google Sheet
The app pulls data from a Google Sheet for each uploaded image. Ensure that the image titles in the folder or the uploaded file match the titles in the Google Drive. The information related to the cashew nut image, such as size, weight, and other metrics, should be available in the sheet.

## Troubleshooting
- If the app doesn't run due to missing dependencies, ensure you have installed all required packages.
- If you encounter issues with missing data in the Google Sheet, verify that the image titles are correctly formatted and match the data in the sheet.

## Key Updates:
1. **Clarified the steps** for running the app.
2. **Corrected the usage of the `streamlit run app.py` command**.
3. Included **clear instructions for the image naming convention** and **matching data in the Google Sheet**.
4. Added **basic troubleshooting** tips for common issues that might arise when running the app.
   

Let me know if you'd like to add or modify anything further!
