# Load libraries
library(shiny)
library(shinydashboard)
library(tidyverse)
library(tidytext)
library(DT)

# Load the data (assuming it's already cleaned and in the correct format)
data <- read.csv("NEAS_CALL_DATA.csv", fileEncoding = "UTF-8")

# Clean and preprocess the data (same steps as before)
data_cleaned <- data %>%
  mutate(
    Impressions = tolower(Impressions),
    Impressions = removePunctuation(Impressions),
    Impressions = removeNumbers(Impressions),
    Impressions = removeWords(Impressions, stopwords("en")),
    Impressions = str_squish(Impressions),
    Impressions = str_replace_all(Impressions, "[^[:print:]]", ""),
    Nature.of.Call = str_trim(Nature.of.Call),
    Symptom.Group = str_trim(Symptom.Group),
    Report.Symptom.Discriminator = str_trim(Report.Symptom.Discriminator)
  )

# Tokenize the "Impressions" column
tokens <- data_cleaned %>%
  unnest_tokens(word, Impressions)

ui <- dashboardPage(
  dashboardHeader(title = "Ambulance Call Analysis"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Overview", tabName = "overview", icon = icon("bar-chart")),
      menuItem("Text Analysis", tabName = "text_analysis", icon = icon("file-text"))
    )
  ),
  
  dashboardBody(
    tabItems(
      tabItem(
        tabName = "overview",
        fluidRow(
          box(title = "Most Prevalent Nature of Call", width = 6, solidHeader = TRUE,
              plotOutput("nature_of_call_plot")),
          box(title = "Most Prevalent Symptom Group", width = 6, solidHeader = TRUE,
              plotOutput("symptom_group_plot"))
        ),
        fluidRow(
          box(title = "Distribution of NEWS Scores", width = 6, solidHeader = TRUE,
              plotOutput("news_scores_plot")),
          box(title = "Most Prevalent Symptom Group Discriminator", width = 6, solidHeader = TRUE,
              plotOutput("symptom_discriminator_plot"))
        )
      ),
      
      tabItem(
        tabName = "text_analysis",
        fluidRow(
          box(title = "Tokenized Impressions", width = 12, solidHeader = TRUE,
              DTOutput("word_table"))
        ),
        fluidRow(
          box(title = "Filters", width = 4, solidHeader = TRUE,
              selectInput("nature_of_call_filter", "Nature of Call", choices = unique(data_cleaned$Nature.of.Call), selected = NULL),
              selectInput("symptom_group_filter", "Symptom Group", choices = unique(data_cleaned$Symptom.Group), selected = NULL),
              selectInput("discriminator_filter", "Symptom Discriminator", choices = unique(data_cleaned$Report.Symptom.Discriminator), selected = NULL)
          ))
      )
    )
  )
)

server <- function(input, output) {
  
  # Overview: Most Prevalent Nature of Call Plot
  output$nature_of_call_plot <- renderPlot({
    nature_call_dist <- data_cleaned %>%
      group_by(Nature.of.Call) %>%
      summarise(count = n()) %>%
      arrange(desc(count))
    
    ggplot(nature_call_dist, aes(x = reorder(Nature.of.Call, count), y = count)) +
      geom_bar(stat = "identity") +
      labs(title = "Most Prevalent Nature of Call") +
      coord_flip()
  })
  
  # Overview: Most Prevalent Symptom Group Plot
  output$symptom_group_plot <- renderPlot({
    symptom_group_dist <- data_cleaned %>%
      group_by(Symptom.Group) %>%
      summarise(count = n()) %>%
      arrange(desc(count))
    
    ggplot(symptom_group_dist, aes(x = reorder(Symptom.Group, count), y = count)) +
      geom_bar(stat = "identity") +
      labs(title = "Most Prevalent Symptom Group") +
      coord_flip()
  })
  
  # Overview: Distribution of NEWS Scores
  output$news_scores_plot <- renderPlot({
    ggplot(data_cleaned, aes(x = First.NEWS.Score)) +
      geom_histogram(bins = 30, fill = "lightblue", color = "black") +
      labs(title = "Distribution of First NEWS Score")
  })
  
  # Overview: Symptom Discriminator Plot
  output$symptom_discriminator_plot <- renderPlot({
    symptom_discriminator_dist <- data_cleaned %>%
      group_by(Report.Symptom.Discriminator) %>%
      summarise(count = n()) %>%
      arrange(desc(count))
    
    ggplot(symptom_discriminator_dist, aes(x = reorder(Report.Symptom.Discriminator, count), y = count)) +
      geom_bar(stat = "identity") +
      labs(title = "Most Prevalent Symptom Discriminator") +
      coord_flip()
  })
  
  # Text Analysis: Tokenized Words Table (Filtered by user selections)
  filtered_tokens <- reactive({
    data_filtered <- data_cleaned
    
    # Apply filters based on user input
    if (!is.null(input$nature_of_call_filter)) {
      data_filtered <- data_filtered %>%
        filter(Nature.of.Call == input$nature_of_call_filter)
    }
    if (!is.null(input$symptom_group_filter)) {
      data_filtered <- data_filtered %>%
        filter(Symptom.Group == input$symptom_group_filter)
    }
    if (!is.null(input$discriminator_filter)) {
      data_filtered <- data_filtered %>%
        filter(Report.Symptom.Discriminator == input$discriminator_filter)
    }
    
    tokens_filtered <- data_filtered %>%
      unnest_tokens(word, Impressions) %>%
      count(word, sort = TRUE)
    
    return(tokens_filtered)
  })
  
  # Display the tokenized words table
  output$word_table <- renderDT({
    datatable(filtered_tokens())
  })
  
}

shinyApp(ui = ui, server = server)
