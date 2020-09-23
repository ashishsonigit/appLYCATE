


# -------------------------------------------
# Welcome tab
# -------------------------------------------


ui <- dashboardPage(
  
  dashboardHeader(title = "Sales Monitoring AI"),

  dashboardSidebar(
    sidebarMenu(
      menuItem("Background", tabName = "Background", icon = icon("dashboard")),
      menuItem("Insights", tabName = "Insights", icon = icon("database")),
      menuItem("Model", tabName = "Model", icon = icon("android"))
    )
  ),

  dashboardBody(
    tabItems(
      # First tab content
      tabItem(tabName = "Background",
              tabPanel(
                'Welcome',
                fluidPage(
                  titlePanel("Overview "),
                  fluidRow(
                    slickROutput("slickr",width = "100%", height = "600px")
                  )
                )
              )

      ),
      # Second tab content
      tabItem(tabName = "Insights",
              tabPanel(
                'Case File View',
                fluidPage(
                  titlePanel(""),
                  fluidRow(
                    column(width=12,
                           tabBox(title='', width = NULL, side = "left", selected = 'Analyse',
                                tags$head(
                                  tags$style(type='text/css',".nav-tabs {font-size: 15px} ")),

                                  tabPanel('Data',

                                              fluidRow(
                                                column(8,align="center",
                                                       box(title='',width = NULL,
                                                        plotlyOutput("CaseCount",width = "100%", height = "400px")
                                                       )
                                                      ),
                                                tags$hr(),
                                                column(8,
                                                  h3('Advantages:'),
                                                  tags$ul(
                                                    tags$li('SME verified synthetic data used to develop Sales Standard (SS2) Monitoring tool Machine Learning framework '),
                                                    tags$li('Data & Model created serves as Accelerator : Augment with client existing data to conduct quick PoC')
                                                  )

                                                )
                                              )

                                  ),
                                #tabpanel
                                  tabPanel('Model',
                                         sidebarLayout(
                                           sidebarPanel(width = 3,
                                             fluidRow(
                                               box(title='Filter', width = NULL,
                                        #         radioButtons("FilterCaseLabel", "Cases :",
                                        #                      choices = list("All" = "All",
                                        #                        "compliant" = "compliant",
                                        #                        "non-compliant" = "non-compliant"
                                        #                        ),
                                        #                       selected = "All"
                                        #          ),

                                        #         uiOutput("case_deselect"),
                                        #         actionButton("OK", "Deselect"),
                                                 verbatimTextOutput("value")
                                               ),
                                               box(title='Select', width = NULL,
                                                 selectInput(
                                                       "CaseSelector",
                                                       "Case :",
                                                       choices = c(1:114),
                                                       selected=114)

                                               ),
                                               box(title = "Predict", width = NULL,
                                                   selectInput("Data_PredictUsingModel_SelectAlgoType", "Select Algorithm Type:",width = NULL,
                                                               choices = mdls,selected = mdls[6],multiple=F),
                                                   actionButton('btn_applyModel',label = 'Apply Model',
                                                                icon = icon('bullseye'),#'bullseye','rocket'
                                                                # class='btn-danger fa-lg',
                                                                width='100%')
                                               )

                                               ) # fluidrow
                                             ), #sidebarpanel

                                           mainPanel(
                                            tabBox(title='', width = NULL, side = "left", selected = 'Text',
                                               tabPanel('Text',
                                                      fluidPage(
                                                        fluidRow(
                                                          column(
                                                            width = 12,
                                                            fluidRow(
                                                              box( title = "Verified", solidHeader = TRUE,
                                                               shinydashboard::infoBoxOutput("ActualLabelInfoBox")),
                                                              box( title = "Prediction", solidHeader = TRUE,
                                                                shinydashboard::infoBoxOutput("PredictedLabelInfoBox"))
                                                            ),
                                                            fluidRow(
                                                              box(
                                                                width = 12,
                                                                shiny::tags$head(shiny::tags$style(shiny::HTML(
                                                                  "#text { font-size: 18px; height: 500px; overflow: auto; }"
                                                                ))),
                                                                textOutput("text")
                                                              )#box
                                                            )#fluidrow
                                                          ) #column
                                                        ) # fluidrow
                                                       ) # fluidpage
                                                    ),
                                                   tabPanel('wordcloud',
                                                      wordcloud2Output("word_cloud",width = "800", height = "580")
                                                      ),
                                                   tabPanel('Word Frequency',
                                                      plotlyOutput("WordFreqChart")

                                                   ) # tabpanel
                                                 ) #tabbox
                                              ) # mainpanel
                                         ) # sidebarlayout
                                  ), #tabpanel

                                tabPanel('Analyse',
                                      sidebarLayout(
                                        sidebarPanel(width = 3,
                                           box(width = NULL,
                                             radioButtons("inputType", "Input Type",
                                                          choices = list("Speech" = "Speech",
                                                                         "Transcript" = "File" #,
                                                                       #  "Text" = "Text"
                                                          ),
                                                          selected = "File"
                                             ),
                                             conditionalPanel(condition = "input.inputType == 'Speech'",
                                               selectInput("TranscribeService",'Transcribe service', 
                                                           choices = list("AWS" = "AWS",
                                                                          "NTE" = "NTE"
                                                           ),
                                                           selected = "AWS", 
                                                           width="150px")
                                             )
                                            ),
                                           box(width = NULL,

                                             radioButtons("NewCaseStatus", "Verify",
                                                          choices = list("compliant" = "compliant",
                                                                         "non-compliant" = "non-compliant",
                                                                         "Not Verified" ="Not Verified"
                                                          ),
                                                          selected = "Not Verified"
                                             )
                                           ),
                                           box(width = NULL,

                                             shinyDirButton("savedir", "Save folder", "Select Save folder"),
                                             verbatimTextOutput("txt_file", placeholder = TRUE)
                                           )
                                        ),
                                        mainPanel(
                                         fluidPage(
                                          box( title = "Verify Status", solidHeader = TRUE,
                                                shinydashboard::infoBoxOutput("ActualNewCaseLabelInfoBox")),

                                          box( title = "Prediction", solidHeader = TRUE,
                                                shinydashboard::infoBoxOutput("PredictedNewCaseLabelInfoBox")),
                                          
                                          

                                          tags$hr(),

                                          conditionalPanel(condition = "input.inputType == 'Text'",

                                             HTML('<textarea id="newcase_textarea" rows="20" cols="100" background-color = "lightyellow">Enter text here</textarea>')
                                            ),
                                          conditionalPanel(condition = "input.inputType == 'File'",
                                             hr(),
                                             br(),         
                                             tabBox(title='', width = NULL, side = "left", selected = 'Text',
                                              tabPanel('Text',
                                                fluidRow(
                                                 column(width=12,
                                                      shinyFilesButton("Btn_GetFile", "Choose a file" ,
                                                                      title = "Please select a file:", multiple = FALSE,
                                                                      buttonType = "default", class = NULL),
                                                      br(),
                                                      hr(),
                                                     box(
                                                       width = 12,
                                                       shiny::tags$head(shiny::tags$style(shiny::HTML(
                                                         "#text { font-size: 18px; height: 500px; overflow: auto; }"
                                                       ))),
                                                       htmlOutput("textFromFile"),
                                                       br(),
                                                       hr()
                                                     )#box
                                                    )#column
                                                   )#fluidrow
                                                  ),#tabpanel,
                                              tabPanel('WordCloud',
                                                 #conditionalPanel(condition = "output.textFromFile!='Text Empty ..'||output.textFromFile!=''",
                                                   wordcloud2Output("word_cloud_File_from_text_to_features",width = "800", height = "580")
                                                 #)                                                 
                                              ),
                                              tabPanel('Interpret',
                                                 #Introduce resutls from LIME
                                                 htmlOutput("lime_predict_interpret"),
                                                 h5('Note:'),
                                                 tags$ul(
                                                   tags$li("Above are the 10 most important features and their ranges that influence the chosen model's decision for this particular case"),
                                                   tags$li('Float point number on the horizontal bars represent the relative importance of these features')
                                                 )
                                               )#tabpanel 
                                              )#tabsetpanel
                                           ),#ConditionalPanel File,
                                          conditionalPanel(condition = "input.inputType == 'Speech'",
                                          
                                            verbatimTextOutput("console"),
                                            hr(),
                                            fluidRow(
                                              column(3,
                                               conditionalPanel(
                                                condition = ("output.console=='Select audio file' || output.console=='Transcription Complete with status : COMPLETED' || output.console=='Transcription Complete with status : TRANSCRIBED'"),
                                                  bsTooltip("Btn_GetaudioFile", "Select audio file for transcription", 
                                                            placement = "bottom", trigger = "hover",
                                                            options = NULL),
                                                  shinyFilesButton("Btn_GetaudioFile", 
                                                                    "Select file",
                                                                    title = "Select audio file", 
                                                                    multiple = FALSE,
                                                                    buttonType = "default", 
                                                                    class = NULL)
                                                 ),
                                                conditionalPanel( 
                                                 condition = ("output.console!='Select audio file' && output.console!='Transcription Complete with status : COMPLETED' && output.console!='Transcription Complete with status : TRANSCRIBED' "),
                                                    bsTooltip("Transcribe", "Click on Transcribe. Please wait while the audio file gets transcribed", 
                                                             placement = "bottom", trigger = "hover",
                                                             options = NULL),
                                                    #h5('Click on Transcribe. Please wait while the audio file gets uploaded to AWS S3 storage and transcribed'),
                                                    actionButton("Transcribe", "Transcribe")
                                                )
                                               ),
                                              column(4, offset=1,
                                                conditionalPanel( 
                                                   condition = ("output.console!='Select audio file' && output.console!='Transcription Complete with status : COMPLETED' && output.console!='Transcription Complete with status : TRANSCRIBED' "),
                                                   img(src = 'spinner2.gif', height = '50px', width = '50px')
                                                )#conditional panel 
                                              ) #column
                                             ), # fluidrow
                                          hr(),
                                          br(),         
                                          tabBox(title='', width = NULL, side = "left", selected = 'Transcribed Text',
                                                 tabPanel('Transcribed Text',
                                                    fluidRow(
                                                      column(12,
                                                         #h5('Transcribed Text'),
                                                         box(#status = "primary",
                                                           width = 12,
                                                           status="warning",solidHeader=FALSE,
                                                           shiny::tags$head(shiny::tags$style(shiny::HTML(
                                                             "#text { font-size: 18px; height: 500px; overflow: auto; }"
                                                           ))),
                                                           
                                                           htmlOutput('transcribedText') # text with newline characters output   textOutput()
                                                           
                                                         )#box Transcribed Text
                                                      )#column
                                                    )#fluidrow
                                                 ),#tabpanel
                                                 tabPanel('Word Cloud',
                                                    conditionalPanel(condition = "output.transcribedText!='Text Empty ..'",
                                                     wordcloud2Output("word_cloud_from_text_to_features",width = "800", height = "580")
                                                    )                                                 
                                                  )
                                                 
                                          )#tabBox
                                         ),#conditionalPanel Speech

                                         actionButton('PredictNewCase_btn',label = 'Predict'),
                                         #  icon("cog", lib = "glyphicon")),
                                         actionButton("submit", "submit"),

                                         actionButton("Clear", "Reset")
                                        )#fluidPage,
                                      )#MainPanel
                                    )#sidebarlayout
                                )
                           ) # tabbox
                     )# column
                  )#fluidrow
                )#fluidpage
              )
      ),
      # Third tab content
      tabItem(tabName = "Model",
                fluidPage(
                  titlePanel("Explorer"),
                      tabBox(title='', height = "900", width = NULL, side = "left", selected = 'ML Features',
                         tags$head(
                            tags$style(type='text/css',".nav-tabs {font-size: 15px} ")),
                        tabPanel('ML Features',

                          #----- Sub-TabPanels
                          tabBox(title='', width = NULL, side = "left", selected = 'Features',
                                 tabPanel('Features',
                                          fluidPage(
                                            fluidRow(
                                              box( title = "Feature Table", width = "12",solidHeader = T,
                                                   DT::dataTableOutput("feature_table"),
                                                   style = "height:400px; overflow-y: scroll;overflow-x: scroll;"
                                              )
                                            )
                                          )
                                 ),
                                 tabPanel('Correlation',
                                          fluidPage(
                                            fluidRow(
                                               box( title = "Correlation Plot",solidHeader = T,
                                                    plotlyOutput('corr_plot')
                                               )
                                            )
                                          )
                                 ),
                                 tabPanel('Visualize Dimensional Data',
                                          fluidPage(
                                            fluidRow(
                                              box( title = "",solidHeader = T,
                                                   plotlyOutput('tsne_xy_plot'),
                                                   p("About t-SNE:"),
                                                   tags$ul(
                                                     tags$li('t-Distributed Stochastic Neighbor Embedding (t-SNE) is a technique for dimensionality reduction that is particularly well suited for the visualization of high-dimensional datasets. ')
                                                   )
                                              ),
                                              box(title = "",solidHeader = T,
                                                   plotlyOutput('pca_xy_plot'),
                                                   p("About PCA:"),
                                                   tags$ul(
                                                     tags$li('Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components. ')
                                                   )
                                               )

                                            )
                                          )
                                 ),

                                 tabPanel('XY Plots',
                                          sidebarLayout(

                                            sidebarPanel(width =3,
                                                         #box(title = "XY Plots: Select inputs", status = "primary",solidHeader = T,
                                                         selectInput("x_feature", "X Feature:",
                                                                     choices=names(ml_df),selected = "case"),

                                                         selectInput("y_feature", "Y Feature:",
                                                                     choices=names(ml_df),selected = "n_differ"),

                                                         box(title='Bubble Size', width = NULL,

                                                             selectInput("bubble_feature", "Select Feature:",
                                                                         choices=names(ml_df),selected = "n_break"),

                                                             sliderInput("scale_add", label = "Coarse Adjust", min = 10,
                                                                         max = 40, value = 40),

                                                             sliderInput("scale_exp", label = "Fine Adjust", min = 1,
                                                                         max = 10, value = 2)


                                                         ) #box

                                            ), # sidepanel

                                            mainPanel(
                                              plotlyOutput('XY_plot')


                                              )

                                            ) # sidebarlayout


                                   ),#tabpanel

                                 tabPanel('Density Plots',

                                          sidebarLayout(

                                            sidebarPanel(width =3,

                                                         selectInput("density_feature", "Select Feature:",
                                                                     choices=names(ml_df),selected = "case")

                                            ), # sidebarpanel

                                            mainPanel(
                                              plotlyOutput('density_plot')

                                            ) # mainPanel

                                          ) # End of sidebarlayout
                                 )# End of sub-tabPanels
                           ) #tabbox
                          #----- end of tabPanel
                          ),
                        tabPanel('Model',
                          tabBox(title='', width = NULL, side = "left", selected = 'Model Training',
                            tabPanel('Model Training',
                                 sidebarLayout(
                                   sidebarPanel(width =3,
                                    fluidPage(
                                      fluidRow(
                                             box(title = 'Model Options',solidHeader = T,status = 'primary',width = NULL,

                                                 selectInput('scoring_metric',label = 'Metric:'%>%label.help('lbl_algo_metric'),
                                                             choices = scoring_metric,selected = scoring_metric,multiple=F),

                                                 radioButtons('rdo_CVtype',label = 'Cross-validation folds'%>%label.help('lbl_CV'),
                                                              choices = c('3-fold'=3,'5-fold'=5,'10-fold'=10),inline = T),

                                                 actionButton('btn_train',label = 'Train Models',
                                                              icon = icon('cogs'),#'bullseye','rocket'
                                                             # class='btn-danger fa-lg',
                                                              width='100%'),
                                                 bsTooltip(id = "lbl_algo", title = "Which algorithms to test",
                                                           placement = "right", trigger = "hover"),
                                                 bsTooltip(id = "lbl_algo_metric", title = "Model performance metric",
                                                           placement = "right", trigger = "hover"),
                                                 bsTooltip(id = "lbl_CV", title = "Number of splits of training data used to tune parameters",
                                                           placement = "right", trigger = "hover")
                                            ),
                                        box(title = 'Filter Results',solidHeader = T,width =NULL, status = 'primary',
                                                 selectInput('resultType',label='Result Type:',
                                                             choices = resultType,selected = resultType,multiple=F),

                                                 selectInput('filter_modelname',label='Select Algorithms:',
                                                             choices = mdls,selected = mdls[6],multiple=T)
                                            )
                                       )
                                     ) # fluidPage
                                   ), # sidebarpanel
                                   mainPanel(
                                     tabBox(title='', width = NULL, side = "left", selected = 'Model Compare',
                                            tabPanel(title = 'Model Statistics',
                                               fluidRow(
                                                 h4('Model Performance Statistics and Tune Parameters'),
                                                 selectInput('ViewModelParam_SelectAlgoType',label='Algorithm Type:',width = '50%',
                                                             choices = unique(CVtune_init$estimator),selected = "RandomForestClassifier",multiple=F),
                                                 tags$hr(),
                                                 p(strong("Score :")),
                                                 tableOutput("models_info_table"),
                                                 tags$hr(),
                                                 selectInput('ViewModelParam_SelectAlgo',label='Algorithm:',width = '50%',
                                                             choices = CVtune_init$ModelName ,selected = "RandomForestClassifier",multiple=F)
                                                 ,
                                                 p(strong("Parameters :")),
                                                 tableOutput("model_parameters_table")
                                               )
                                            ),
                                            tabPanel(title = 'Model Compare',#icon = icon('sort-amount-asc'),
                                             fluidRow(
                                               h4('Cross-validation results:'),
                                               plotOutput('CVplot1',height=600)
                                              )
                                            ),
                                            tabPanel(title = 'Observed vs Predicted',
                                                     h4('Observed vs Predicted (best candidate for algorithm)'),
                                                     selectInput('Select_X_Or_x_test',label='Validate model on:',width = '30%',
                                                                 choices = list('All Data'='All Data',
                                                                                'Validation set' = 'Validation'),
                                                                 selected = 'All Data',multiple=F),
                                                     plotOutput('CVplot2',height=600)
                                            )
                                   ) # tabBox
                                 ) # mainPanel
                              ) # End of sidebarlayout
                            )# End of tabPanel 'Model Training
                         )# End of tabBox select tabPanel'Model Training'
                       )# End of tabPanel Model
                     )# End of tabbox
                   )#End of fluid page
                ) # End of tabItem


  ) #End  of tabItems
 ) # end of Dashbordbody
) #dashboardpage


