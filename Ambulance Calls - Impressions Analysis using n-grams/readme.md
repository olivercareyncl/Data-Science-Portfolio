
# Ambulance Call Analysis Shiny App

## Description
The **Ambulance Call Analysis Shiny App** is an interactive application built using R's Shiny framework to analyse ambulance call data. This app utilises natural language processing (NLP) techniques to perform text analysis on the 'Impressions' column, which contains free-text notes taken by first responders at the scene of an emergency. The app allows users to filter the data by various parameters, such as the nature of the call, symptom group, and symptom discriminator, and perform N-gram analysis to extract possible insights from these impression notes. 

The purpose of this app is to provide insights into the ambulance call records by analysing the unstructured text data, helping to identify common patterns, trends, and anomalies in the responses to emergency situations.

## Technologies Used
- **Shiny**: R framework for building interactive web applications.
- **R**: The programming language used for data analysis and app development.
- **DT**: Library for rendering interactive tables in the app.
- **tm**: Text mining package for text preprocessing.
- **tidyverse**: Collection of R packages used for data manipulation and visualization.
- **tidytext**: Package for text mining and NLP in R.

## Installation and Setup
To run this app locally, follow these steps:

1. **Install R**: Ensure that you have R installed. If not, you can download and install it from [CRAN](https://cran.r-project.org/).
2. **Install RStudio**: If you don’t have RStudio installed, you can download it from [RStudio](https://rstudio.com/products/rstudio/download/).
3. **Install Required Packages**: Open RStudio and run the following commands to install the necessary R packages:

```r
install.packages(c("shiny", "shinydashboard", "tidyverse", "tidytext", "DT", "tm"))
```

4. **Clone the Repository**: Clone this repository to your local machine using the following command:
   
```bash
git clone https://github.com/olivercareyncl/Data-Science-Portfolio/Ambulance Calls - Impressions Analysis using n-grams/ambulance-call-analysis.git
```

5. **Run the App**: After installing the dependencies and cloning the repository, open the app directory in RStudio and run the app by executing:

```r
shiny::runApp()
```

The app should open in your default web browser.

## How to Use the App
Once the app is running, you can interact with it as follows:

1. **Select Filters**: Use the dropdown menus to select specific filters based on the `Nature of Call`, `Symptom Group`, or `Symptom Discriminator`. You can select multiple options for each filter.
2. **N-gram Analysis**: After applying the filters, specify the size of the N-grams (e.g., bi-grams, tri-grams) to examine the most frequent word combinations in the `Impressions` text.
3. **View Results**: The filtered dataset will be displayed in a table, showing the `Call ID`, `Impressions`, `Nature of Call`, `Symptom Group`, and `Symptom Discriminator` columns.
4. **Download Filtered Data**: You can download the filtered data as a CSV file for further analysis by clicking the "Download Filtered Data as CSV" button.
5. **Clear Filters**: Click the "Clear All Filters" button to reset the filter selections and start the analysis over.

## Features

### Data Filtering Options
- **Nature of Call**: Filter data based on the urgency of the call (e.g., "Emergency", "Routine", "Urgent").
- **Symptom Group**: Filter by different categories such as "Cardiac", "Respiratory", "Neurological", and more.
- **Symptom Discriminator**: Filter by the severity of the symptoms (e.g., "Severe", "Moderate", "Mild").

### N-gram Analysis
- Users can specify the size of N-grams (e.g., bi-grams or tri-grams) to analyze frequently occurring word combinations in the `Impressions` column.
- This feature helps identify patterns in the text data, such as common phrases or terminology used by first responders.

### Data Download
- Users can download the filtered dataset as a CSV file for further offline analysis.

## Simulated Data
The app uses a simulated dataset containing 1,000 rows of ambulance call data. The dataset includes the following columns:

- **Call.ID**: A unique identifier for each call.
- **Impressions**: Free-text notes taken by first responders at the scene. This column contains non-standardized, unstructured text such as "Chest pain, patient sweating and pale, possible heart attack".
- **Nature.of.Call**: The nature of the call (e.g., "Emergency", "Routine", "Urgent").
- **Symptom.Group**: A classification of the symptoms observed (e.g., "Cardiac", "Respiratory", "Neurological").
- **Report.Symptom.Discriminator**: A classification of the severity of symptoms (e.g., "Severe", "Moderate", "Mild").

The `Impressions` column is intentionally left as free text to simulate real-world data entries made by first responders.

## Contribution
If you would like to contribute to this project, feel free to fork the repository, make your changes, and submit a pull request. All contributions are welcome!

### Guidelines for Contribution:
- Ensure that the app functions correctly after any changes.
- Follow best practices for R and Shiny app development.
- Include appropriate comments and documentation for your code.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

---

Thank you for checking out the **Ambulance Call Analysis Shiny App**! If you have any questions or suggestions, feel free to open an issue or contact me directly.

