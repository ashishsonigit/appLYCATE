


server <- function(input, output, session) {
  
  output$VariableImportance <- renderPlotly({
    len = 20
    p <- plot_ly(data = py$df[1:len,], x = ~Word, y = ~Count, type = "bar", color = ~Word, height = 500, width = 200)
    p <- layout(p, xaxis = list(categoryarray = ~Word, categoryorder = "array"),showlegend = FALSE)
    p
  })
  
  output$CaseCount <- renderPlotly({
    data = clean_df
    freq <-data.frame(table(data$label))
    plot_ly(data = freq, x = ~Var1, y = ~Freq, type = "bar", color = ~Var1) %>%
      layout(title = paste('Total Cases : ',nrow(data)),
             xaxis = list(title = 'Case Type'),
             yaxis = list(title = ' Count '))
  })
  
  output$WordFreqChart <- renderPlotly({
    
    dat <-wordFreqTable(data(),'text',input$CaseSelector)
    p <- plot_ly(data = dat, x = ~Var1, y = ~Freq, type = "bar", color = ~Var1)%>%
      layout(title = 'Word Count',
             xaxis = list(title = 'Word'),
             yaxis = list(title = 'Count'))
    p <- layout(p, xaxis = list(categoryarray = ~Var1, categoryorder = "array"),showlegend = FALSE)
    p
  })
  
  output$corr_plot <- renderPlotly({
    # calculate correlations
    temp<-ml_df[,2:(length(ml_df)-1)]
    correlations <- cor(temp)
    Names <- names(temp)
    xaxes_format <- list(
      title = "Features",
      showticklabels = TRUE,
      tickangle = 45
    )
    yaxes_format <- list(
      title = "Features",
      showticklabels = TRUE
    )
    
    plot_ly(z = correlations,x=Names,y=Names,type = "heatmap") %>%
      layout(xaxis = xaxes_format, yaxis = yaxes_format,  width=800,
             height=600,showlegend = FALSE)
    
  })
  
  output$corr_3Dplot <- renderPlotly({
    # calculate correlations
    temp<-ml_df[,2:(length(ml_df)-1)]
    correlations <- cor(temp)
    Names <- names(temp)
    xaxes_format <- list(
      title = "Features"
    )
    yaxes_format <- list(
      title = "Features"
    )
    
    plot_ly(z = correlations)%>% #colorscale = "Hot",x=Names,y=Names,type = "heatmap"
      add_surface() %>%
      layout(xaxis = xaxes_format, yaxis = yaxes_format, showlegend = FALSE)
  })
  
  output$event <- renderPrint({
    d <- event_data("plotly_hover")
    if (is.null(d)) "Hover on a point!" else d
  })
  
  
  values <- reactiveValues(data =  clean_df)
  
  # observeEvent(input$FilterCaseLabel, {
  #  data = clean_df
  #  lbl = isolate(input$FilterCaseLabel)
  #  if (lbl != "All") {
  #    values$data <- data[data$label == lbl,]
  #  }
  # })
  
  # observeEvent(input$OK, {
  #  deselect = strtoi(input$caseDeSelector)
  #  temp<- values$data
  #  temp <- temp[temp$case != c(deselect),]
  #  values$data <- temp
  # })
  
  data <- reactive({
    
    data <- clean_df
    
    #lbl<- isolate(input$FilterCaseLabel)
    
    #if (lbl != "All") {
    #  data <- data[data$label == lbl,]
    #}
    data
  })
  
  
  output$feature_table <- renderDataTable({
    
    datatable(ml_df, options = list(paging = FALSE,rownames = FALSE))
    
  })
  
  output$feature_stats_table <- renderDataTable({
    
    # Descriptive statistics for each column
    datatable(summary(ml_df), options = list(paging = FALSE,rownames = FALSE))
    
    
  })
  
  #Plot for Trace Explorer
  output$XY_plot <- renderPlotly({
    
    #Use the ideal sizeref value
    desired_maximum_marker_size <- input$scale_add
    list_of_size_values <- ml_df[ ,input$bubble_feature]
    sizeref <- input$scale_exp * max(list_of_size_values) / (desired_maximum_marker_size**2)
    my_ml_df <- ml_df
    my_ml_df$size_values<-list_of_size_values
    p <- plot_ly(data = my_ml_df,
                 x = my_ml_df[ ,input$x_feature],
                 y = my_ml_df[ ,input$y_feature],
                 color = ~label,
                 type = 'scatter',
                 mode = 'markers',
                 hovermode = "closest",
                 text = paste("Size :", my_ml_df$size_values,
                              ", label :", my_ml_df$label,
                              ", case :", my_ml_df$case
                 ),
                 hoverinfo = "text",
                 marker = list(size = list_of_size_values, opacity = 0.6,sizemode = 'area', sizeref = sizeref)
    ) %>%
      layout(title = ' XY Plot',
             xaxis = list(title = input$x_feature),
             yaxis = list(title = input$y_feature),showlegend = TRUE)
    p
  })
  
  
  #Scatter Plot - tsne
  output$tsne_xy_plot <- renderPlotly({
    
    #Use the ideal sizeref value
    desired_maximum_marker_size <- 10
    list_of_size_values <- ml_df[ ,'n_remaining']
    sizeref <- input$scale_exp * max(list_of_size_values) / (desired_maximum_marker_size**2)
    
    seed<-10
    df_tsne <- as.data.frame(X_tsne)
    df_tsne$label<-ml_df$label
    df_tsne$case<-ml_df$case
    df_tsne$size_values<-list_of_size_values
    
    p <- plot_ly(data = df_tsne,
                 x = df_tsne$V1,
                 y = df_tsne$V2,
                 color = ~label,
                 type = 'scatter',
                 mode = 'markers',
                 hovermode = "closest",
                 text = paste("case :", df_tsne$case,
                              "label :", df_tsne$label),
                 hoverinfo = "text",
                 marker = list(opacity = 0.6,sizemode = 'area')  #size = list_of_size_values,  sizeref = sizeref
    ) %>%
      layout(title = ' t-SNE : Scatter Plot',
             xaxis = list(title = 'V1'),
             yaxis = list(title = 'V2'),showlegend = TRUE)
    #p <- layout(p, xaxis = list(categoryarray = ~input$x_feature, categoryorder = "array"),showlegend = FALSE)
    p
  })
  
  #Scatter Plot - tsne
  output$pca_xy_plot <- renderPlotly({
    
    
    #Use the ideal sizeref value
    desired_maximum_marker_size <- 10
    list_of_size_values <- ml_df[ ,'n_remaining']
    sizeref <- input$scale_exp * max(list_of_size_values) / (desired_maximum_marker_size**2)
    
    seed<-10
    df_pca <- as.data.frame(X_pca)
    df_pca$label<-ml_df$label
    df_pca$case<-ml_df$case
    df_pca$size_values<-list_of_size_values
    
    p <- plot_ly(data = df_pca,
                 x = df_pca$V1,
                 y = df_pca$V2,
                 color = ~label,
                 type = 'scatter',
                 mode = 'markers',
                 hovermode = "closest",
                 text = paste("case :", df_pca$case,
                              "label :", df_pca$label),
                 hoverinfo = "text",
                 marker = list(opacity = 0.6,sizemode = 'area')  #size = list_of_size_values,  sizeref = sizeref
    ) %>%
      layout(title = ' PCA : Scatter Plot',
             xaxis = list(title = 'V1'),
             yaxis = list(title = 'V2'),showlegend = TRUE)
    p
  })
  
  output$density_plot <- renderPlotly({
    
    ml_df1 <- ml_df[which(ml_df$label == "compliant"),]
    density1 <- density(ml_df1[ ,input$density_feature])
    
    ml_df2 <- ml_df[which(ml_df$label == "non-compliant"),]
    density2 <- density(ml_df2[ ,input$density_feature])
    
    p <- plot_ly(x = ~density1$x, y = ~density1$y, name = 'compliant',type = 'scatter', mode = 'lines',   fill = 'tozeroy',
                 fillcolor = 'rgba(168, 216, 234, 0.5)',
                 line = list(width = 0.5)) %>%
      add_trace(x = ~density2$x, y = ~density2$y, name = 'non-compliant', fill = 'tozeroy',
                fillcolor = 'rgba(255, 212, 96, 0.5)') %>%
      layout(xaxis = list(title = input$density_feature),
             yaxis = list(title = 'Density'),showlegend = TRUE)
  })
  
  observe({
    tmp <- values$data
    updateSelectInput(session, "CaseSelector",
                      choices = tmp$case
    )
  })
  
  output$case_deselect<-renderUI({
    tmp <- values$data
    selectInput("caseDeSelector","Exclude Cases:", choices =tmp$case, multiple = TRUE,selected = NULL)
    
  })
  
  output$value <- renderText({ input$case_deselect })
  
  output$word_cloud = renderWordcloud2(
    DispWordCloud(values$data,'text',input$CaseSelector)
  )
  
  output$word_cloud_from_text_to_features = renderWordcloud2(
    DispWordCloudFromText(raw$Text)
  )
  
  output$word_cloud_File_from_text_to_features = renderWordcloud2(
    DispWordCloudFromText(raw_file_input$Text)
  )
  
  output$text <- renderText({
    dat = values$data
    req(input$CaseSelector)
    as.character(dat[dat$case == input$CaseSelector,][, 'text', drop = FALSE])
  })
  
  observeEvent(input$submit,{
    
    dat$Newcase = 0
    casefolder = dirinfo$selected
    casefolder = paste0(casefolder,'/outputs/scored/',as.character(as.POSIXlt(Sys.time()), format = "%m%d%y-%H%M"))
    
    dir.create(casefolder, showWarnings = FALSE, recursive = TRUE, mode = "0777")
    
    casefile = paste0(casefolder,"/maincorpus.txt",sep="")
    statusfile = paste0(casefolder,"/label.txt",sep="")
    
    if(input$inputType == 'Text'){
      cat(paste(input$newcase_textarea), file=casefile)
      cat(paste(input$NewCaseStatus), file=statusfile)
      updateTextInput(session, "newcase_textarea", value = "Enter text here")
    }
    
    if(input$inputType == 'Speech'){
      
      cat(paste(raw$Text), file=casefile)
      cat(paste(input$NewCaseStatus), file=statusfile)
    }
    
    if(input$inputType == 'File'){
      cat(paste(FromFile$text), file=casefile)
      cat(paste(input$NewCaseStatus), file=statusfile)
    }
    
  }
  )
  
  
  
  # Reactive Values
  
  FromFile <- reactiveValues(text =  "Text Empty ..")
  
  raw_file_input <- reactiveValues(Text =  "Text Empty ..")
  
  raw <- reactiveValues(Text = "Text Empty ..",RecordedFileName = "",RecordedFilePath=PATH_VOICESAMPLES, transcribe_status = "")
  
  
  dat <- reactiveValues(models_chosen = models_chosen_init,CVtune=CVtune_init,CVgrid=CVgrid_init,SelectedEstimatorName="GradientBoostingClassifier",Newcase=0,NewcasePredictButtonPressed =0,NewcaseCurrInputype="Text")
  
  dirinfo<-reactiveValues(selected = PATH)
  
  # Save Directory
  shinyDirChoose(input, "savedir",
                 roots = c(main = PATH,
                           k_drive = K_DRIVE),
                 defaultRoot = "main",
                 session=session)
  
  observeEvent(input$savedir, {
    dirinfo$selected <- parseDirPath(PATH, input$savedir)
    output$txt_file <- renderText(dirinfo$selected)
  })
  
  
  output$slickr <- renderSlickR({
    imgs <- list.files("./background", pattern=".png", full.names = TRUE)
    slickR(obj=imgs)
  })
  
  observeEvent(input$btn_train,{
    
    shinyjs::disable('btn_train')
    on.exit(shinyjs::enable('btn_train'))
    
    #mdls <<- isolate(input$slt_algo)
    
    
    dat$models_chosen <<- as.data.frame(unlist(mdls))[,1]
    
    scorer = switch(input$scoring_metric,
                    'accuracy' = accuracy_scorer,
                    'recall'= recall_scorer,
                    'roc_auc' = 'roc_auc',
                    'f1'= f1_scorer
    )
    option = list(cv = as.integer(as.integer(input$rdo_CVtype)),scoring=scorer)
    
    CV_res<-cvmodel_tuning(x_train,y_train,option)
    dat$CVtune <-CV_res$tune
    dat$CVgrid <-CV_res$grid
    
    
  })
  
  CVres <- reactive({
    
    CVres<-dat$CVtune%>%filter(estimator %in% dat$models_chosen)
    
    CVres
  })
  
  
  
  topModels <- reactive({
    
    if(is.null(CVres()))
      return()
    
    ModelName_list <- CVres()%>%filter(estimator %in% input$ViewModelParam_SelectAlgoType)%>%select(ModelName) #,mean_train_score,std_train_score,mean_test_score,std_test_score)
    
    lst<-as.character(as.data.frame(ModelName_list)[,1])
    
    lst
  })
  
  
  observe({
    lst <- topModels()
    updateSelectizeInput(session,'ViewModelParam_SelectAlgo',choices = lst,selected = lst[1])
  })
  
  
  output$models_info_table <- renderTable({
    
    df<- dat$CVtune%>%filter(estimator %in% input$ViewModelParam_SelectAlgoType)%>%
      #select(ModelName, mean_train_score,std_train_score,mean_test_score,std_test_score)%>%
      select(ModelName, mean_test_score,std_test_score)%>%
      arrange(desc(mean_test_score))

    df1<-df%>%select(mean_test_score,std_test_score)
    
    row.names(df1)<-df$ModelName
  
    
    df1
    
  },rownames = TRUE,align = 'l')
  
  output$model_parameters_table <- renderTable({
    
    df<-modelParam(CVres(),modelNamestr=input$ViewModelParam_SelectAlgo)
    
    df
    
  },rownames = TRUE,align = 'l')
  
  output$CVplot1 <- renderPlot({
    
   
    
    
   # resdf = switch(input$resultType,
   #                 'train' = {
   #                  CVres()%>%mutate(min_score = mean_train_score-std_train_score,
   #                                  max_score = mean_train_score+std_train_score,
   #                                  mean_score = mean_train_score
   #                  )%>%select(ModelName,estimator,min_score,max_score,mean_score)
   #                 },
   #                 'test'= {
   #                  CVres()%>%mutate(
   #                     min_score = mean_test_score-std_test_score,
   #                     max_score = mean_test_score+std_test_score,
   #                     mean_score = mean_test_score
   #                   )%>%select(ModelName,estimator,min_score,max_score,mean_score)
   #                }
   #  )
    
    resdf <- CVres()%>%mutate(min_score = mean_test_score-std_test_score,
                              max_score = mean_test_score+std_test_score,
                              mean_score = mean_test_score
                            )%>%select(ModelName,estimator,min_score,max_score,mean_score)
    
    resdf<-resdf%>%filter(estimator %in% input$filter_modelname)
    
    xlabel_str = switch(input$resultType,
                        'train' = " Train Score",
                        'test'  =  " Test Score"
    )
    
    pal <- wes_palette('Darjeeling1',n = length(unique(y_train)),type = 'c')
    ggplot(resdf,aes(x=ModelName,color=estimator))+
      geom_errorbar(aes(ymin=min_score,ymax=max_score),size=1)+
      geom_point(aes(y=mean_score),size=3)+
      #scale_color_manual(values=pal)+
      coord_flip()+
      theme_bw()+
      ylab(xlabel_str)+
      theme(legend.position='none') -> p2
    
    gridExtra::grid.arrange(p2,ncol=1)
  })
  
  
  
  output$Classification_Report <- renderTable({
    if(input$Select_X_Or_x_test=='All Data'){
      obs= Y
      xdata=X
    }else{
      obs=y_test
      xdata=x_test
    }
    pred=eval(parse(text=paste('dat$CVgrid$grid_searches$',dat$SelectedEstimatorName,'$best_estimator_$predict(xdata)',sep='')))
    report<-classificationReport(obs,pred)
    report
  })
  
  
  output$CVplot2 <- renderPlot({
    if(input$Select_X_Or_x_test=='All Data'){
      obs= Y
      xdata=X
    }else{
      obs=y_test
      xdata=x_test
    }
    df=as.data.frame(c())
    
    str=as.character(models_chosen_init)
    getObsPred <- function(i){
      EstimatorName=str[i]
      pred=eval(parse(text=paste('dat$CVgrid$grid_searches$',EstimatorName,'$best_estimator_$predict(',deparse(substitute(xdata)),')',sep='')))
      name = EstimatorName
      df <- data.frame(name,obs,pred)
      
      
    }
    df <- plyr::ldply(1:length(str),getObsPred)
    
    pal <- wes_palette('Darjeeling1',n = length(unique(y_train)),type = 'c')
    df %>% group_by(pred,obs,name) %>%
      summarise(n=n()) %>%
      ggplot(.)+
      geom_raster(aes(x=obs,y=pred,fill=name,alpha=n))+
      geom_text(aes(x=obs,y=pred,label=n))+
      # scale_fill_manual(values=pal)+
      coord_equal()+
      facet_wrap(~name)+
      theme_bw()+
      xlab('Observed')+
      ylab('Predicted')+
      theme(legend.position='none')
    
  })
  
  
  observeEvent(input$btn_applyModel,{
    
    shinyjs::disable('btn_applyModel')
    on.exit(shinyjs::enable('btn_applyModel'))
    
    dat$SelectedEstimatorName = input$Data_PredictUsingModel_SelectAlgoType
    
  })
  
  
  
  output$ActualLabelInfoBox <- renderInfoBox({
    
    dat = data()
    req(input$CaseSelector)
    # req(exists(input$CaseSelector, "package:datasets", inherits = FALSE), cancelOutput = TRUE)
    
    lhs = as.character(dat[dat$case == input$CaseSelector,][, c('label'), drop = FALSE][1,])
    
    icon_label  = ifelse(lhs == 'non-compliant', "thumbs-down fa", "thumbs-up fa")
    
    infoBox(
      '', tags$p('', style = "font-size: 65%;"), icon = icon(icon_label, lib = "glyphicon"), #Actual
      color = ifelse (lhs == 'non-compliant', "orange", "aqua")
    )
  })
  
  
  
  output$PredictedLabelInfoBox <- renderInfoBox({
    
    
    i=input$CaseSelector
    
    
    lhs=eval(parse(text=paste('dat$CVgrid$grid_searches$',
                              dat$SelectedEstimatorName,
                              '$best_estimator_$predict(',deparse(substitute(X)),'[as.integer(','i','),])',
                              sep='')))
    
    
    probab=eval(parse(text=paste('dat$CVgrid$grid_searches$',
                                 dat$SelectedEstimatorName,
                                 '$best_estimator_$predict_proba(',deparse(substitute(X)),'[as.integer(','i','),])',
                                 sep='')))
    
    icon_label  = ifelse(lhs == 'non-compliant', 
                         "thumbs-down fa", 
                         ifelse(lhs == 'compliant',
                                "thumbs-up fa",
                                'exclamation-triangle'))
    
    valuetext   = ifelse(lhs == 'non-compliant',
                         paste0('Score ',round(probab[,2],2),sep=''),
                         ifelse(lhs == 'compliant',
                                paste0('Score ',round(probab[,1],2),sep=''),
                                'Not available')
    )
    COLOR  = ifelse(lhs == 'non-compliant', 
                    "orange", 
                    ifelse(lhs == 'compliant',
                           "aqua",
                           'orange'))
    
    infoBox(
      '',tags$p(valuetext, style = "font-size: 90%;") , icon = icon(icon_label, lib = "glyphicon"),  #Predicted
      color = COLOR,width=12
    )
  })
  
  
  
  output$gauge1 <- flexdashboard::renderGauge({
    lhs <- predicted_compliance()$type
    probab <- predicted_compliance()$type_prob[,2]*100
    percent   = ifelse(lhs == 'non-compliant', round(probab), (100-round(probab)))
    flexdashboard::gauge(percent, min = 0, max = 100, symbol = '%', label = paste("Probability"),
                         flexdashboard::gaugeSectors(success = c(65, 100), warning = c(50,65), danger = c(0, 50), color = ifelse (lhs == 'non-compliant', "orange", "aqua")
                         )
    )
  })
  
  
  
  
  
  
  output$ActualNewCaseLabelInfoBox <- renderInfoBox({
    
    req(input$NewCaseStatus)
    
    
    icon_label  = ifelse(input$NewCaseStatus == 'non-compliant', "thumbs-down fa",
                         ifelse(input$NewCaseStatus == 'compliant',"thumbs-up fa","exclamation-triangle"))
    
    
    color_type = ifelse (input$NewCaseStatus == 'non-compliant', "orange", 
                         ifelse(input$NewCaseStatus == 'compliant', "aqua", "orange"))
    
    ICON = icon(icon_label)  
    
    
    infoBox(
      '', tags$p(input$NewCaseStatus, style = "font-size: 90%;"), icon = ICON,    #Verify Status
      color = color_type
    )
  })
  
  Global_Mlmodel <- reactiveValues(text="",model =  "",Lime_interpret="")
  
  getPage<-function() {
    return(includeHTML(LIME_EXPLANATION))
  }
  output$lime_predict_interpret<-renderUI({getPage()})
  
  
  
  output$limeImage <- renderImage({
    # A temp file to save the output.
    # This file will be removed later by renderImage
    outfile <- tempfile(fileext = 'lime.png')
    
    # Generate the PNG
    png(outfile, width = 400, height = 300)
    hist(rnorm(input$obs), main = "Generated in renderImage()")
    dev.off()
    
    # Return a list containing the filename
    list(src = outfile,
         contentType = 'image/png',
         width = 400,
         height = 300,
         alt = "This is alternate text")
  }, deleteFile = TRUE)
  
  output$PredictedNewCaseLabelInfoBox <- renderInfoBox({
    
    MLmodel = eval(parse(text=paste('dat$CVgrid$grid_searches$',
                                    dat$SelectedEstimatorName,
                                    '$best_estimator_',sep='')))
    
    
    if(input$inputType == 'Text'){
      text = isolate(input$newcase_textarea)
    }else if(input$inputType == 'File'){
      text = paste(FromFile$text,collapse=" ")
    }else if(input$inputType == 'Speech'){
      text = raw$Text
    }

    
    #Lime_interpret = model_interpret_new_text(text,'unknown',cases,MLmodel,10)
    #exp1.show_in_notebook(show_table=True)
    
    if(resetAll$reset == TRUE){
      lhs=''
    }else{
      
      #writeLines(text, "../outputs/outfile.txt")
      NewCasePred_df= generate_prediction_FromText(text,MLmodel)
      lhs = NewCasePred_df$PredictedLabel
      probab = round(NewCasePred_df$PredictedProbability_NC,2)
      Global_Mlmodel$model = MLmodel
      Global_Mlmodel$text = text
      
      
      # Global_MLmodel$Lime_interpret = Lime_interpret
      
    }
    
    
    
    GlobalValueTextPredicted$icon_label  = ifelse(lhs == 'non-compliant', "thumbs-down fa",
                                                  ifelse(lhs == 'compliant',"thumbs-up fa","exclamation-triangle"))
    
    
    GlobalValueTextPredicted$Valuetext = ifelse(lhs == 'non-compliant', paste0('Score ',probab,sep=''),
                                                ifelse(lhs == 'compliant', paste0('Score ',1-probab,sep=''),
                                                       paste0('Score NA')))
    
    color_type = ifelse (lhs == 'non-compliant', "orange", 
                         ifelse(lhs == 'compliant', "aqua", "orange"))
    
    ICON = icon(GlobalValueTextPredicted$icon_label)                                     
    
    
    infoBox(
      '',tags$p(GlobalValueTextPredicted$Valuetext, style = "font-size: 90%;") , icon =ICON,   #Prediction
      color = color_type,width=12
    )
    
  })    
  
  
  
  observeEvent(input$PredictNewCase_btn,{
    
    print("observeEvent Predict")
    shinyjs::disable('PredictNewCase_tbn')
    on.exit(shinyjs::enable('PredictNewCase_btn'))
    
    dat$Newcase = 1
    dat$NewcasePredictButtonPressed = 1
    resetAll$reset = FALSE
    
    
  })
  
  
  observeEvent(input$inputType,{
    print("observeEvent inputtype")
    output$console <-renderText({"Select audio file"})
    if(dat$NewcaseCurrInputype != input$inputType){
      updateRadioButtons(session, "NewCaseStatus",
                         choices = list("compliant" = "compliant",
                                        "non-compliant" = "non-compliant",
                                        "Not Verified" ="Not Verified"
                         ),
                         selected = "Not Verified"
      )
      resetAll$reset= TRUE
      raw$Text <- ""
      updateTextInput(session, "newcase_textarea", value = "Enter text here")
    }
  })
  
  
  
  GlobalValueTextPredicted <- reactiveValues(Valuetext =  "Not available",icon_label="exclamation-triangle")
  
  
  observeEvent(input$inputaudioType,{
    if(input$inputaudioType == "Microphone"){
      output$console <-renderText({"Start speaking into Microphone...Recording stops when silence period > 3 sec"})
    }else{
      output$console <-renderText({"Select audio file"})
    }
  })
  
  
  Global_audioFile<- reactiveValues(enable = TRUE)
  
  shinyFileChoose(input, "Btn_GetaudioFile",
                  roots = c(main = PATH,
                            k_drive = K_DRIVE),
                  filetypes=c('wav'),
                  session = session)
  
  observeEvent(input$Btn_GetaudioFile,{
    
    audioFile_selected<-parseFilePaths(roots = c(main = PATH,
                                                 k_drive = K_DRIVE), input$Btn_GetaudioFile)
    FILE <- paste(as.character(audioFile_selected$datapath), collapse="\n")
    Global_audioFile$FILE = FILE
    raw$RecordedFilePath <- dirname(FILE)
    raw$RecordedFileName <- basename(FILE)
    resetAll$reset= TRUE
    
    output$transcribedText <- renderText({
      "Text Empty .."
    })
    output$console <-renderText({paste0("audio File : ",FILE)})
    Global_audioFile$enable = FALSE
    
  })
  
  output$audiofile_actionbutton_flag<-reactive({
    return(isTRUE(Global_audioFile$enable))
  })
  
  outputOptions(output, "audiofile_actionbutton_flag", suspendWhenHidden = FALSE)
  
  observe({
    delay(1, toggleState("Transcribe", condition = ((is.null(raw$RecordedFileName) || raw$RecordedFileName == "") || raw$transcribe_status !="")))
  })
  
  observeEvent(input$Transcribe,{
    
    print("observeEvent Transcribe")
    Global_audioFile$enable = TRUE
    
    
    FILE_DIR = raw$RecordedFilePath
    FILE_NAME = raw$RecordedFileName
    option=input$TranscribeService
    print(option)
    resetAll$reset = TRUE
    job_result = py$TranscribeRecordedSpeech(FILE_DIR,FILE_NAME,option)
    multicolortext = multicolor_highlight(
      job_result[[1]]
    )
    output$transcribedText<-renderText({
      paste(multicolortext, sep = "", collapse = '<br/>')
    })
    
    raw$Text = job_result[[1]]
    raw$transcribe_status = job_result[[2]][1]
    output$console <-renderText({paste0("Transcription Complete with status : ",raw$transcribe_status)})
    
    
  })
  
  
  
  resetAll <- reactiveValues(reset = TRUE)
  
  output$transcribedText <- renderText({
    "Text Empty .."
  })
  
  
  observeEvent(input$Clear,{
    print("Entering Clear")
    resetAll$reset= TRUE
    output$console <-renderText({"Select audio file"})
    
    output$transcribedText <- renderText({
      "Text Empty .."
    })
    output$textFromFile<-renderText({
      "Text Empty .."
    })
    
    updateRadioButtons(session, "NewCaseStatus",
                       choices = list("compliant" = "compliant",
                                      "non-compliant" = "non-compliant",
                                      "Not Verified" ="Not Verified"
                       ),
                       selected = "Not Verified"
    )
    
    
    
    raw$Text <- "Text Empty .."
    raw_file_input$Text<-"Text Empty .."
  })
  
  
  shinyFileChoose(input, "Btn_GetFile",
                  roots = c(main = PATH,
                            k_drive = K_DRIVE),
                  filetypes=c('txt'),
                  session = session)
  
  # Simple highlight strings 
  simple_highlight <- function(text) {
    keywords <- c("spk0", "spk1", "spk2")
    x <- unlist(strsplit(text, split = ": ", fixed = T))
    x[tolower(x) %in% tolower(keywords[0])] <- paste0("<mark>", x[tolower(x) %in% tolower(keywords)], "</mark>")
    paste(x, collapse = " ")
  }
  
  
  # Mark with different colors
  multicolor_highlight <- function(list_text) {
    keywords <- c("spk_0:", "spk_1:", "spk_2:")
    color<-c("cyan","orange","yellow")
    list_text = unlist(strsplit(list_text, split = "\n", fixed = T))
    for(k in 1:length(list_text)){
      text = list_text[k]
      x <- unlist(strsplit(text, split = " ", fixed = T))
      for(i in 1:length(keywords)){
        x[tolower(x) %in% tolower(keywords[i])] <- paste0("<span style='background-color:",color[i],"'>", x[tolower(x) %in% tolower(keywords[i])],"</span>")
        
      }
      list_text[k] = paste(x, collapse = " ")
      
    }
    return(list_text)
  }
  
  multicolor_highlight_textfile <- function(list_text) {
    keywords <- c("spk_0:", "spk_1:", "spk_2:")
    color<-c("cyan","orange","yellow")
    #list_text = unlist(strsplit(list_text, split = "\n", fixed = T))
    for(k in 1:length(list_text)){
      text = list_text[k]
      x <- unlist(strsplit(text, split = " ", fixed = T))
      for(i in 1:length(keywords)){
        x[tolower(x) %in% tolower(keywords[i])] <- paste0("<span style='background-color:",color[i],"'>", x[tolower(x) %in% tolower(keywords[i])],"</span>")
        
      }
      list_text[k] = paste(x, collapse = " ")
      
    }
    return(list_text)
  }
  
  
  
  
  observeEvent(input$Btn_GetFile,{
    
    file_selected<-parseFilePaths(roots = c(main = PATH,
                                            k_drive = K_DRIVE), input$Btn_GetFile)
    FromFile$text <- readLines(paste(as.character(file_selected$datapath), collapse="\n"))
    
    raw_file_input$Text = paste(FromFile$text, collapse=' ' )
    
    multicolortext = multicolor_highlight_textfile(
      FromFile$text
    )
    
    
    resetAll$reset= TRUE
    # print(multicolortext)
    output$textFromFile<-renderText({
      #renderUI({
      #HTML(
      paste(multicolortext, sep = "", collapse = '<br/>')
      #)
      
    })
    output$transcribedText <- renderText({
      "Text Empty .."
    })
    
  })
}





