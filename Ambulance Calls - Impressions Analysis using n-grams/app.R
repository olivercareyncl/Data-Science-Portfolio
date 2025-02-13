library(shiny)
library(shinydashboard)
library(tidyverse)
library(tidytext)
library(DT)
library(tm)

# Simulate a dataset with non-standardized free-text 'Impressions' column
set.seed(123)

# Generate simulated data
data_simulated <- tibble(
  Call.ID = 1:1000,
  Impressions = sample(
    c(
      "Patient unconscious, not responding to stimuli",
      "Chest pain, patient sweating and pale, possible heart attack",
      "Breathing difficulties, pain in chest, dizziness",
      "Severe head injury, patient bleeding from forehead",
      "Patient has a fever and shortness of breath, possible infection",
      "Unresponsive to verbal cues, shallow breathing",
      "Multiple lacerations on the arms, bleeding",
      "Panic attack, hyperventilating, chest tightness",
      "Patient complaining of severe back pain, unable to move",
      "Fell from a height, complaining of leg pain",
      "Possible stroke, slurred speech, one side of face drooping",
      "Patient complaining of severe stomach cramps and nausea",
      "Traumatic injury to right ankle, swelling visible",
      "Patient has fainted, no apparent injuries",
      "Heavy bleeding from nose, breathing normally",
      "Burns on hands, possible chemical exposure",
      "Patient in shock, low blood pressure",
      "Chest tightness and wheezing, possible asthma attack",
      "Severe headache, possible migraine or concussion",
      "Patient confused, disoriented, possible intoxication"
    ),
    1000, replace = TRUE
  ),
  Nature.of.Call = sample(c("Emergency", "Routine", "Urgent"), 1000, replace = TRUE),
  Symptom.Group = sample(c("Cardiac", "Respiratory", "Neurological", "Trauma", "Gastrointestinal", "Psychiatric"), 1000, replace = TRUE),
  Report.Symptom.Discriminator = sample(c("Severe", "Moderate", "Mild", "Critical"), 1000, replace = TRUE)
)

# Clean and preprocess the simulated data
data_cleaned <- data_simulated %>%
  mutate(
    Impressions = tolower(Impressions),  # Convert to lowercase
    Impressions = removePunctuation(Impressions),  # Remove punctuation
    Impressions = removeNumbers(Impressions),  # Remove numbers
    Impressions = removeWords(Impressions, stopwords("en")),  # Remove stopwords
    Impressions = str_squish(Impressions),  # Remove extra spaces
    Impressions = str_replace_all(Impressions, "[^[:print:]]", ""),  # Remove non-printable characters
    Nature.of.Call = str_trim(Nature.of.Call),  # Trim whitespace
    Symptom.Group = str_trim(Symptom.Group),  # Trim whitespace
    Report.Symptom.Discriminator = str_trim(Report.Symptom.Discriminator)  # Trim whitespace
  )

ui <- dashboardPage(
  dashboardHeader(title = "NEAS Impressions"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("App Introduction", tabName = "app_intro", icon = icon("info-circle")),
      menuItem("N-Gram Analysis", tabName = "text_analysis", icon = icon("file-text"))
    )
  ),
  
  dashboardBody(
    tabItems(
      # Introduction Tab
      tabItem(
        tabName = "app_intro",
        fluidRow(
          box(title = "Welcome to Ambulance Impressions NLP Analysis App", width = 12, solidHeader = TRUE, status = "primary",
              h3("App Overview"), 
              p("This Shiny app uses natural language processing techniques to analyse simulated ambulance call data, specifically focusing on 'Impressions' from call records. The goal is to help explore key insights into the nature of calls, symptoms, and their relationships to the text-based data in the 'Impressions' column."),
              h4("Author: Oliver Carey"),
              p("This app was developed by Oliver Carey as part of a project for North East Ambulance Service. This version has been run using simulation data to be presented as a portfolio piece.")
          )
        )
      ),
      
      # Impressions Analysis Tab
      tabItem(
        tabName = "text_analysis",
        
        # Narrative and Instructions for Impressions Analysis
        fluidRow(
          box(title = "Impressions Analysis Overview", width = 12, solidHeader = TRUE, status = "primary",
              h4(tags$strong("Overview")),
              p("This section allows you to analyse the 'Impressions' column from the simulated ambulance call dataset. You can filter the dataset based on various factors such as the nature of the call, symptom group, and symptom discriminator. This enables a more focused analysis of the 'Impressions' text data."),
              p("You can perform the following steps in this section:"),
              tags$ol(
                tags$li("Use the filters to select specific categories based on the nature of the call, symptom group, or symptom discriminator."),
                tags$li("After applying the filters, specify the size of the N-grams (e.g., bi-grams, tri-grams) to examine the most frequent word combinations from the 'Impressions' text."),
                tags$li("View the filtered dataset in a table that includes call ID, impressions, and additional call information. You can search for specific n-grams of interest."),
                tags$li("Download the filtered data as a CSV file for further analysis."),
                tags$li("Clear filters to reset the view and start over.")
              )
          )
        ),
        
        # Filters Section
        fluidRow(
          box(title = "Filters", width = 12, solidHeader = TRUE,
              p("You can apply multiple filters to narrow down the dataset for analysis."),
              selectInput("nature_of_call_filter", "Nature of Call", choices = unique(data_cleaned$Nature.of.Call), selected = NULL, multiple = TRUE),
              actionButton("add_nature_filter", "Add Filter for Nature of Call"),
              br(),
              selectInput("symptom_group_filter", "Symptom Group", choices = unique(data_cleaned$Symptom.Group), selected = NULL, multiple = TRUE),
              actionButton("add_symptom_group_filter", "Add Filter for Symptom Group"),
              br(),
              selectInput("discriminator_filter", "Symptom Discriminator", choices = unique(data_cleaned$Report.Symptom.Discriminator), selected = NULL, multiple = TRUE),
              actionButton("add_discriminator_filter", "Add Filter for Symptom Discriminator"),
              br(),
              textInput("ngram_size", "Enter n for N-grams Analysis (e.g., 2 for bi-grams)", value = "2"),
              actionButton("clear_all_filters", "Clear All Filters"),
              p("Once filters are applied, the analysis results will be updated automatically.")
          )
        ),
        
        # Active Filters Display
        fluidRow(
          box(title = "Active Filters", width = 12, solidHeader = TRUE,
              p("Currently applied filters are displayed below. You can remove any filter by clicking the corresponding 'Remove' button."),
              uiOutput("active_filters_ui")
          )
        ),
        
        # N-gram Analysis Section
        fluidRow(
          box(title = "N-gram Analysis Results", width = 12, solidHeader = TRUE,
              p("N-grams are combinations of words that frequently appear together in the 'Impressions' column. You can adjust the N-gram size to explore bi-grams, tri-grams, etc."),
              DTOutput("ngram_table")
          )
        ),
        
        # Call ID Table with Impressions and other Filters
        fluidRow(
          box(title = "Filtered Call Data", width = 12, solidHeader = TRUE,
              p("This table displays the filtered dataset based on the selected filters. It includes the `Call ID`, `Impressions`, `Nature of Call`, `Symptom Group`, and `Symptom Discriminator` columns. Use the search box to look up specific n-grams in 'Impressions' found in previous section."),
              DTOutput("callid_table"),
              downloadButton("download_csv", "Download Filtered Data as CSV")
          )
        )
      )
    )
  )
)

server <- function(input, output, session) {
  
  # Reactive values to store active filters
  filters <- reactiveValues(nature_of_call = NULL, symptom_group = NULL, discriminator = NULL)
  
  # Add filters for text analysis
  observeEvent(input$add_nature_filter, {
    if (!is.null(input$nature_of_call_filter)) {
      filters$nature_of_call <- unique(c(filters$nature_of_call, input$nature_of_call_filter))
    }
  })
  
  observeEvent(input$add_symptom_group_filter, {
    if (!is.null(input$symptom_group_filter)) {
      filters$symptom_group <- unique(c(filters$symptom_group, input$symptom_group_filter))
    }
  })
  
  observeEvent(input$add_discriminator_filter, {
    if (!is.null(input$discriminator_filter)) {
      filters$discriminator <- unique(c(filters$discriminator, input$discriminator_filter))
    }
  })
  
  # Clear all filters
  observeEvent(input$clear_all_filters, {
    filters$nature_of_call <- NULL
    filters$symptom_group <- NULL
    filters$discriminator <- NULL
  })
  
  # Filter UI rendering for active filters
  output$active_filters_ui <- renderUI({
    active_filters <- list()
    if (!is.null(filters$nature_of_call)) {
      active_filters <- append(active_filters, 
                               list(tags$div(paste("Nature of Call: ", paste(filters$nature_of_call, collapse = ", ")), 
                                             actionButton("remove_nature_filter", "Remove")))
      )
    }
    if (!is.null(filters$symptom_group)) {
      active_filters <- append(active_filters, 
                               list(tags$div(paste("Symptom Group: ", paste(filters$symptom_group, collapse = ", ")),
                                             actionButton("remove_symptom_group_filter", "Remove")))
      )
    }
    if (!is.null(filters$discriminator)) {
      active_filters <- append(active_filters, 
                               list(tags$div(paste("Symptom Discriminator: ", paste(filters$discriminator, collapse = ", ")),
                                             actionButton("remove_discriminator_filter", "Remove")))
      )
    }
    do.call(tagList, active_filters)
  })
  
  # Remove filters
  observeEvent(input$remove_nature_filter, {
    filters$nature_of_call <- NULL
  })
  
  observeEvent(input$remove_symptom_group_filter, {
    filters$symptom_group <- NULL
  })
  
  observeEvent(input$remove_discriminator_filter, {
    filters$discriminator <- NULL
  })
  
  # Filtered data for text analysis
  filtered_data <- reactive({
    data_filtered <- data_cleaned
    
    if (!is.null(filters$nature_of_call)) {
      data_filtered <- data_filtered %>%
        filter(Nature.of.Call %in% filters$nature_of_call)
    }
    
    if (!is.null(filters$symptom_group)) {
      data_filtered <- data_filtered %>%
        filter(Symptom.Group %in% filters$symptom_group)
    }
    
    if (!is.null(filters$discriminator)) {
      data_filtered <- data_filtered %>%
        filter(Report.Symptom.Discriminator %in% filters$discriminator)
    }
    
    return(data_filtered)
  })
  
  # N-gram Analysis Table
  output$ngram_table <- renderDT({
    ngram_size <- as.numeric(input$ngram_size)
    
    if (is.na(ngram_size) || ngram_size <= 0) {
      return(NULL)  # Invalid n-gram size
    }
    
    ngrams_filtered <- filtered_data() %>%
      unnest_tokens(ngram, Impressions, token = "ngrams", n = ngram_size) %>%
      filter(!str_detect(ngram, "^\\s*$")) %>%  # Exclude blank n-grams
      count(ngram, sort = TRUE)
    
    datatable(ngrams_filtered)
  })
  
  # Call ID Table with Impressions, Nature of Call, Symptom Group, and Symptom Discriminator
  output$callid_table <- renderDT({
    # Extract unique Call IDs along with the additional fields
    callid_data <- filtered_data() %>%
      select(Call.ID, Impressions, Nature.of.Call, Symptom.Group, Report.Symptom.Discriminator) %>%
      distinct()  # Get distinct rows
    
    datatable(callid_data)
  })
  
  # Download CSV handler for the Call ID table
  output$download_csv <- downloadHandler(
    filename = function() {
      paste("callid_data_", Sys.Date(), ".csv", sep = "")
    },
    content = function(file) {
      callid_data <- filtered_data() %>%
        select(Call.ID, Impressions, Nature.of.Call, Symptom.Group, Report.Symptom.Discriminator) %>%
        distinct()  # Get distinct rows
      
      write.csv(callid_data, file, row.names = FALSE)
    }
  )
  
}

shinyApp(ui = ui, server = server)

