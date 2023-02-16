# 采用新的贝叶斯模型平均软件ensembleBMA

# bma multimodel, to predict precipitation
rm(list = ls(all=TRUE)) #clear 
#library(BMA)
#library(EBMAforecast)
library(ensembleBMA)

# x=c(1,2,3,4)
# y=c(1,2,3,4)
# s<- EM.normals(x, y)

file_num <- 6

for (num in 1:1)
{
  file= paste('D:\\干旱\\data\\result8\\nmme_qm_preci_', as.character(num), '_raftery.txt', sep='')
  all_models <- read.table(file) #time*(pts*model)
  
  file= 'D:\\干旱\\data\\result8\\gpcc_china_6016_raftery.txt'
  preci <- read.table(file)
  preci[is.na(preci)] <- 0
  preci <- preci[((1982-1960)*12+1):((2016-1960+1)*12), ]
  
  # dim(all_models) <- c(61,36,816,56)
  
  print('read file finished!')
  pts <- 968 # 65*37
  row <- dim(all_models)[1]
  col <- dim(all_models)[2]
  # lead <- 6
  model_num <- col / pts
  all_models_new <- matrix(nrow=row, ncol=pts)
  
  # model_name <- seq(from=1, by=1, to=model_num)
  # model_name <- as.character(model_name)
  # model_name <- t(model_name)
  weight <- matrix(nrow=pts, ncol=model_num)
  
  # all_models <- array(all_models, dim=c(row, pts, model_num) )
  # weight <- array(weight, dim=c(pts, model_num))
  # all_models_new <- array(all_models_new, dim=c(row, pts))
  
  tn_start_yr <- 1982
  tn_end_yr <- 2010
  # tn_start_yr <- 2009
  # tn_end_yr <- 2016
  tt_start_yr <- 2011
  tt_end_yr <- 2016
  
  # date_vector <- seq(from=117001, by=1, to=117001 - 1 + (tt_end_yr-1960+1)*12)
  date_vector <- read.table('D:\\干旱\\data\\result8\\date.txt') #time*(pts*model)
  date_vector <- date_vector[,1]
  date_vector <- as.character(date_vector)
  # char_vector <- character( (tt_end_yr-tn_start_yr+1)*12 )
  # for (i in 1:length(char_vector))
  # {
  #   char_vector[i] <- 117001 + i -1
  # }
  all_models_new_probab <- matrix(nrow = (tt_end_yr - tt_start_yr + 1)*12, ncol=pts*100)
  
  for (i in 1:300)
  {
    x <- all_models[, seq(from=i,by=pts,to=col)] #816*56
    # x <- all_models[ , i, ] #816*56
    x <- data.matrix(x)
    x_new <- x[seq(from=(tn_start_yr-tn_start_yr)*12+1,by=1,to=(tn_end_yr-tn_start_yr+1)*12),]
    x_new <- data.matrix(x_new)
    x_test <- x[seq(from=(tt_start_yr-tn_start_yr)*12+1,by=1,to=(tt_end_yr-tn_start_yr+1)*12),]
    x_test <- data.matrix(x_test)
    y <- preci[, i] #1*168
    y <- data.matrix(y)
    # y <- t(y)
    y_new <- y[seq(from=(tn_start_yr-tn_start_yr)*12+1,by=1,to=(tn_end_yr-tn_start_yr+1)*12), ]
    y_new <- as.numeric(y_new)
    y_test <- y[seq(from=(tt_start_yr-tn_start_yr)*12+1,by=1,to=(tt_end_yr-tn_start_yr+1)*12), ]
    y_test <- as.numeric(y_test)
    if( (sum(y_new)==0) | (sum(x_new)==0) )
    {
      next
    }
    
    # forecast_data <- makeForecastData(.predCalibration = x_new, .outcomeCalibration = y_new, 
    #                                   .predTest = x_test,
    #                                   .outcomeTest = y_test, .modelNames= model_name)
    # result <- calibrateEnsemble(.forecastData = forecast_data,exp=1,tol=sqrt(.Machine$double.eps),
    #                             maxIter = 1e+06,model="normal",method="EM")
    
    col_index <- seq(from=1,by=1,to=length(y_new))
    col_index_date = as.character(col_index)
    # col_index <- setdiff(col_index, preci_miss_index[preci_miss_index<=max(ori_index)])
    # col_index <- setdiff(col_index, 1) # ignore the first value
    # col_index <- setdiff(col_index, preci_miss_index_add1[preci_miss_index_add1 <= max(ori_index)])
    # col_index <- intersect( col_index, preci_index_good[i,])
    noan_index <- !is.na(x_new[1,])
    
    if( sum(noan_index) <=1 )
    {
      next
    }
    
    col_sum <- colSums(x_new)
    noan_index <- col_sum > 1.0
    index_m <- seq(from=1, by=1, to=model_num)
    noan_index <- index_m[noan_index]
    
    if(length(noan_index) <= 1)
    {
      next
    }
    
    
    tempTestData <- ensembleData( forecasts = x[,noan_index],
                                  dates = date_vector[1:length(y)],
                                  observations = y,
                                  station = '1',
                                  forecastHour = 24,
                                  initializationTime = "00")
    # tempTestData <- ensembleData( forecasts = x_new[col_index, noan_index],
    #                               dates = date_vector[1:length(y_new)],
    #                               observations = y_new[col_index],
    #                               station = '1',
    #                               forecastHour = 0,
    #                               initializationTime = "00")
    # model <- ensembleBMAnormal(tempTestData, trainingDays=length(y_new), dates = '20010905',
    #                            control = controlBMAnormal(), exchangeable = NULL,
    #                            minCRPS = FALSE)
    model <- ensembleBMAgamma0( tempTestData, trainingDays = length(y_new),
                                dates= date_vector[(length(y_new)+1):length(y)],
                                control = controlBMAgamma0(maxIter = length(noan_index)*20, tol = 1e-5,
                                                           power = (1/3), rainobs = 10,
                                                           init = list(varCoefs = NULL, weights = NULL),
                                                           optim.control = 1e-5 ))
    # ,dates= date_vector[(length(y_new)+1):length(y)]
    # control = controlBMAgamma0(maxIter = length(noan_index)*50, tol = 1e-3,
    #                            power = (1/3), rainobs = 10,
    #                            init = list(varCoefs = NULL, weights = NULL),
    #                            optim.control = 1e-3 )
    
    tempTestData_2 <- ensembleData( forecasts = x[,noan_index],
                                    dates = date_vector[1:length(y)],
                                    observations = y,
                                    station = '1',
                                    forecastHour = 24,
                                    initializationTime = "00")
    qtile <- seq(from=0.005, by=0.01, to=0.995)
    qforecast <- quantileForecast( model, tempTestData_2, quantiles = qtile, dates=date_vector[(length(y_new)+1):length(y)] )
    
    all_models_new_probab[, ((i-1)*100+1):(i*100)] <- qforecast
    
    
    w <- model$weights
    w <- rowMeans(w)
    # forecast_data <- makeForecastData(.predCalibration = x_new[col_index, noan_index], .outcomeCalibration = y_new[col_index],
    #                                   .predTest = x_test[,noan_index],
    #                                   .outcomeTest = y_test, .modelNames= model_name[noan_index])
    # result <- calibrateEnsemble(.forecastData = forecast_data,exp=3,tol=sqrt(.Machine$double.eps),
    #                             maxIter = 1e+06,model="normal",method="EM")
    
    
    # w <- result@modelWeights
    # w <- as.numeric(w)
    # weight[i, ] <- w
    # all_models_new[ seq(from=i,by=pts,to=row), ] <- x%*%w
    weight[i, noan_index] <- w
    
    
    # all_models_new[ seq(from=i,by=pts,to=row), ] <- y_pre_new
    all_models_new[ , i ] <- x[,noan_index]%*%w
    
    print(i)
    # rmse <- sqrt(mean((all_models_new[ , i ] - y_new) * (all_models_new[ , i ] - y_new)))
    # print(paste0("rmse: ",rmse))
    
    
    
    # 保存每个点的结果
    w_each_point <- w
    bma_each_point <- all_models_new[ , i ]
    qforecast_point <- qforecast
    
    w_each_point[is.na(w_each_point)] <- -999999
    # all_models_new <- array(all_models_new, dim=c(row, pts))
    file2 = paste('//home//xulei//project//climate_dynamics//pts//w_pts_', as.character(i), '_lead_', as.character(num), '_raftery.txt', sep='')
    write.table(w_each_point, file2, quote=FALSE, sep=' ',row.names=FALSE,col.names=FALSE)
    
    bma_each_point[is.na(bma_each_point)] <- -999999
    # all_models_new <- array(all_models_new, dim=c(row, pts))
    file2 = paste('//home//xulei//project//climate_dynamics//pts//bma_pts_', as.character(i), '_lead_', as.character(num), '_raftery.txt', sep='')
    write.table(bma_each_point, file2, quote=FALSE, sep=' ',row.names=FALSE,col.names=FALSE)
    
    
    qforecast_point[is.na(qforecast_point)] <- -999999
    # all_models_new <- array(all_models_new, dim=c(row, pts))
    file2 = paste('//home//xulei//project//climate_dynamics//pts//qforecast_pts_', as.character(i), '_lead_', as.character(num), '_raftery.txt', sep='')
    write.table(qforecast_point, file2, quote=FALSE, sep=' ',row.names=FALSE,col.names=FALSE)
  }
  # all_models_new[is.na(all_models_new)] <- -999999
  # # all_models_new <- array(all_models_new, dim=c(row, pts))
  # file2 = paste('\\home\\swe\\XL\\all_models_bma_', as.character(num), '_raftery.txt', sep='')
  # write.table(all_models_new, file2, quote=FALSE, sep=' ',row.names=FALSE,col.names=FALSE)
  # 
  # weight[is.na(weight)] <- -999999
  # # weight <- array(weight, dim=c(pts, model_num*lead))
  # file2 = paste('\\home\\swe\\XL\\all_models_weight_', as.character(num), '_raftery.txt', sep='')
  # write.table(weight, file2, quote=FALSE, sep=' ',row.names=FALSE,col.names=FALSE)
  
  print(paste('lead:',as.character(num),sep=''))
}


print('bma finished!')

