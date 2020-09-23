library(shiny)
library(shinydashboard)
library(shinycustomloader)
library(shinydashboard)
library(shinyBS)
library(shinyjs)
library(shinyFiles)
library(shinycssloaders)
library(ggplot2)
library(plotly)
library(rjson)
library(flexdashboard)
library(dplyr)
library(DT)
library(wesanderson)
library(reshape2)
library(reticulate)
library(tidytext)
library(wordcloud2)
library(htmltools)
library(htmlwidgets)
library(tm)
library(xtable)
library(slickR)
library(lattice)
library(data.table)
library(corrplot)
library(devtools)
library(yaml)
library(stringr)
library(gridExtra)

config = yaml.load_file('../config.yaml')


use_python(config$paths$PATH_PYTHON)
use_condaenv(config$constants$CONDA_ENV, required = TRUE)

# PATH = config$paths$PATH
# setwd(PATH)
setwd("..")
PATH = getwd()

PYTHON_SOURCEFILE = paste0(PATH,'/',config$paths$PYTHON_SOURCEFILE)
LIME_EXPLANATION  = paste0(PATH,'/',config$paths$LIME_EXPLANATION)
FEATURES          = paste0(PATH,'/',config$paths$FEATURES)
DATA              = paste0(PATH,'/',config$paths$DATA)


K_DRIVE = config$paths$K_DRIVE

KDRIVE = config$paths$KDRIVE
PATH_VOICESAMPLES = paste0(PATH,"data/test")

smt_AWS_ACCESS_KEY_ID     = config$AWS_CONFIG$smt_ACCESS_KEY_ID
smt_AWS_ACCESS_SECRET_KEY = config$AWS_CONFIG$smt_AWS_ACCESS_SECRET_KEY
smt_AWS_REGION            = config$AWS_CONFIG$smt_AWS_REGION
smt_AWS_BUCKET_NAME       = config$AWS_CONFIG$smt_AWS_BUCKET_NAME



dir.create(paste0(PATH,'/','outputs/features'), showWarnings = FALSE, recursive = TRUE, mode = "0777")
dir.create(paste0(PATH,'/','outputs/models'), showWarnings = FALSE, recursive = TRUE, mode = "0777")


# Load python source file
source_python(PYTHON_SOURCEFILE)



clean_df <-doc
clean_df$`Unnamed: 0` <- NULL
clean_df$label<-as.factor(clean_df$label)
clean_df$case<-as.factor(clean_df$case)
data <- clean_df

ml_df<-doc2

wordFreqTable<-function(df,variable,casenumber){
  sentences<-df[casenumber,][, c(variable), drop = FALSE][1,]
  #split sentenses to word tokens
  words<-strsplit(sentences," ")
  #Calculate word frequencies
  words.freq<-table(unlist(words))
  arrange(as.data.frame(words.freq),desc(Freq))
} 


# Display word cloud based on token freque
DispWordCloud<-function(df,variable,casenumber){

  words.freq<-wordFreqTable(df,variable,casenumber)

  wordcloud2(words.freq, size = 0.7) # backgroundColor='black'
} 



# Display word cloud from text based on token frequecy
DispWordCloudFromText<-function(text){

	l<-preprocess_and_generate_features_from_text(text,'unknown',115)
	l<-t(l)
	l[,1]<-as.integer(l[,1])
	wc=data.frame(rownames(l),as.integer(l[,1]))  
	colnames(wc)<-c('features','freq')
	wc <-subset(wc, !(features %in% c('label','case')))
	wc <-wc %>% arrange(desc(freq))
	wordcloud2(wc, size = 0.7)


} 


cvmodel_tuning<-function(x_train,y_train,option){
  

  grid<-model_grid_tuning(x_train,y_train,option)
  
  res = grid$score_summary()

  tune<-res%>% rowwise() %>% mutate(
    ModelName = paste(estimator,paste(unlist(params), collapse='_'),sep="_")
    
  )
  tune$ModelName<-factor(tune$ModelName)
  
  modelnames = tune$estimator%>%unique()
  setwd(PATH)
  for(i in 1:length(modelnames)){
    MLmodelDir  = paste0(getwd(),'/outputs/models/',as.character(as.POSIXlt(Sys.time()), format = "%m%d%y-%H%M%S"))
    dir.create(MLmodelDir, showWarnings = TRUE, recursive = TRUE, mode = "0777")
    
    MLmodelName = paste0(MLmodelDir,'/model_',modelnames[i],'.rds')
    
    MLmodel = eval(parse(text=paste('grid$grid_searches$',modelnames[i],'$best_estimator_',sep='')))

    if(!is.null(MLmodel)){
      print('-----')
      print(MLmodel)
      print(MLmodelName)
      saveRDS(MLmodel, MLmodelName)
      print("file created")
      print('-----')
    }
  }
  
  return(list(tune=tune,grid=grid))
}


CV_init<-cvmodel_tuning(x_train,y_train,option = list(cv = as.integer(5),scoring = accuracy_scorer))
CVtune_init<-CV_init$tune
CVgrid_init<-CV_init$grid




modelParam<-function(df,modelNamestr='RandomForestClassifier_gini_3'){
  
  i=which(df$ModelName==modelNamestr)
  model_param <- as.data.frame(unlist(df$params[i]))
  names(model_param)<-"Value"
  return(model_param)
}


scoring_metric <- list('roc_auc'='roc_auc',
                       'f1' = 'f1',
                       'recall' ='recall',
                       'accuracy'='accuracy')

resultType <- list('test' = 'test')#'train'='train',)

mdls <- list('SVC'='SVC',
             'Random Forest' = 'RandomForestClassifier',
             'SGD'='SGDClassifier',
             'Extra Trees'='ExtraTreesClassifier',
             'Gradient Boosting'='GradientBoostingClassifier',
             'AdaBoost'='AdaBoostClassifier')

# 'SVC Poly'='SVCPoly',
# 'Neural Network'='nnet',
# 'k-NN'='knn',
# 'Naive Bayes'='nb', ,
# 'Logistic Regression' = 'LogisticRegression',


models_chosen_init <<- as.data.frame(unlist(mdls))[,1]

label.help <- function(label,id){
  HTML(paste0(label,actionLink(id,label=NULL,icon=icon('question-circle'))))
}

DisplayConfusionMatrix<-function(EstimatorName,x,y,pal){
  obs = y
  pred=eval(parse(text=paste('CVgrid_init$grid_searches$',EstimatorName,'$best_estimator_$predict(',deparse(substitute(x)),')',sep='')))
  df <- data.frame(obs,pred)
  df$pred <- factor(df$pred,levels=levels(df$obs))
  df %>% group_by(pred,obs) %>% 
    summarise(n=n()) %>% 
    ggplot(.)+
    geom_raster(aes(x=obs,y=pred,alpha=n))+
    geom_text(aes(x=obs,y=pred,label=n))+
    # scale_fill_manual(values=pal)+
    coord_equal()+
    # facet_wrap(~name)+
    theme_bw()+
    xlab('Observed')+
    ylab('Predicted')+
    theme(legend.position='none')
}


