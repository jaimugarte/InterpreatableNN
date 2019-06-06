neural_network_model <-
function(df,vars_info,target,hidden_layers,activation,learn_rate=0.01,regularization=F,
                                 train_percent=0.7,epochs = 50){
  # df: dataset
  # vars_info: vector of length = number of columns of df. 1: numeric ,2: categorical, 3: Date
  # target: name of variable to predict (by default last column)
  # hidden_layers: a vector with the hidden layers
  # activation: "relu","sigmoid","tanh"
  # learn_rate: real number
  # regularization: FALSE or TRUE
  # train_percent: number between 0 and 1 with the percentage of train data between all data
  # epochs: integer with number of ephocs
  
  library(keras)
  library(reticulate)
  library(dplyr)
  
  n_df <- nrow(df)
  p_df <- ncol(df)
  
  # set variables to its class 
  df <- df %>% mutate_if(vars_info == 1, as.numeric)
  df <- df %>% mutate_if(vars_info == 2, as.factor)
  df <- df %>% mutate_if(vars_info == 3, as.Date)
  
  # Transform Date variables to ordered integers
  for(i in 1:p_df){
    if(vars_info[i]==3) {
      aux <- as.Date(unique(df[,i]),origin = "%Y-%m-%d")
      len_aux <- length(aux)
      aux = data.frame(dates = aux[order(aux)], ordered = 1:len_aux)
      aux$dates <- as.character(aux$dates)
      library(plyr)
      new_col <- mapvalues(as.character(df[,i]),
                           from = aux[,"dates"],
                           to = aux[,"ordered"])
      df[,i] = as.numeric(new_col)
      remove(new_col)
    }
  }
  
  # deal with missing values in categorical variables
  cont_NA_cat <- which(lapply(1:p_df, function(i) anyNA(df[,i]) & is.factor(df[,i])) == T)
  for(i in 1:p_df){
    if(i %in% cont_NA_cat){
      df[,i] <- factor(df[,i], levels=unique(c(levels(df[,i]), "NA_value")))
      df[,i] <- replace(df[,i],which(is.na(df[,i])),factor("NA_value"))
    }
  }
  
  # deal with missing values in continuous or countable variables
  cont_NA <- which(lapply(1:p_df, function(i) anyNA(df[,i]) & is.numeric(df[,i])) == T)
  for(i in 1:p_df){
    if(i %in% cont_NA){
      new_var_name <- paste(colnames(df)[i],"_NA",sep = "")
      df[,c(new_var_name)] <- rep(0,n_df)
      df[is.na(df[,i]),c(new_var_name)] <- 1
      df[,c(new_var_name)] <- as.factor(df[,c(new_var_name)])
      df[,i] <- replace(df[,i],which(is.na(df[,i])),0)
    }
  }
  
  # Hot-encoding of categorical variables (except for target variable) 
  cat_vars <- which(vars_info == 2)
  target_col <- which(colnames(df)==target)
  names_new_cols <- c()
  for(i in 1:p_df){
    if(((i %in% cat_vars) & (i!=target_col))==TRUE){
      new_cols <- as.data.frame(to_categorical(unclass(df[,i]))[,-1])
      for(k in 1:ncol(new_cols)) names_new_cols[k] <- paste(colnames(df)[i],levels(df[,i])[k],sep = ".")
      colnames(new_cols) <- names_new_cols
      df <- cbind(df[,-i], new_cols)
    }
  }

  # Divide the different datasets: Sample the dataset
  w <- sample(1:n_df, size = train_percent*n_df)
  
  # Function to scale between 0 and 1 numerical vectors
  range01 <- function(x){           
    if(is.numeric(x)) (x-min(x))/(max(x)-min(x))
    else x} 

  x_train <- as.data.frame(lapply(df[w,-which(colnames(df)==target)[1]], range01))
  x_train <- as.matrix(x_train)
  x_test <- as.data.frame(lapply(df[-w,-which(colnames(df)==target)[1]], range01))
  x_test <- as.matrix(x_test)
  
  if(is.factor(df[,c(target)])==T){
    y_train <- as.matrix(as.integer(unclass(df[w,c(target)])))
    y_test <- as.matrix(as.integer(unclass(df[-w,c(target)])))
    # y_train <- to_categorical(unclass(df[w,c(target)]))
    # y_test <- to_categorical(unclass(df[-w,c(target)]))
  #   y_train <- to_categorical(df[w,c(target)])
  #   y_test <- to_categorical(df[-w,c(target)])
  } else {
    y_train <- df[w,c(target)]
    y_train <- as.matrix(y_train)
    y_test <- df[-w,c(target)]
    y_test <- as.matrix(y_test)
  }
  
  # defining the model and layers
  units_input <- ncol(x_test)    # number of neurons in input layer
  units_output <- ncol(y_test)   # number of neurons in output layer
  
  model <- keras_model_sequential()
  
 # Separate regression problem and classification problem:
  
  if(vars_info[target_col]==1){
    #  REGRESSION PROBLEM 
    hidden_layer_i = 0
    N <- length(hidden_layers)
    
    model %>%
      # Input layer:
      layer_dense(units = hidden_layers[1], activation = activation,
                  input_shape = units_input)
      
      # Hidden layers:
      while(hidden_layer_i < N){
        hidden_layer_i = hidden_layer_i+1
        model %>%
          layer_dropout(rate = 0.4) %>%
          layer_dense(units = hidden_layers[hidden_layer_i], activation = activation)
      }
      
    # Output layer
    model %>%layer_dense(units = units_output, activation = 'relu') %>%
      layer_dropout(rate = 0.4)
    
    # Compile model
    model %>% compile(
      loss = "mse",
      optimizer = optimizer_rmsprop(),
      metrics = list("mean_absolute_error","accuracy")
    )
    
    # Results:
    print_dot_callback <- callback_lambda(
      on_epoch_end = function(epoch, logs) {
        if (epoch %% 80 == 0) cat("\n")
        cat(".")
      }
    )    
    
    # Fit the model and store training stats
    history <- model %>% fit(
      x_train,
      y_train,
      epochs = epochs,
      validation_split = 0.2,
      verbose = 0,
      callbacks = callback_tensorboard()
    )
    library(ggplot2)
    plot(history, metrics = "mean_absolute_error", smooth = FALSE) +
      coord_cartesian(ylim = c(0, 5))
    
  } else {
  #  CLASIFICATION PROBLEM  
    hidden_layer_i = 0
    N <- length(hidden_layers)
    
    model %>%
      # Input layer:
      layer_dense(units = hidden_layers[1], activation = activation,
                  input_shape = units_input)
    
    # Hidden layers:
    while(hidden_layer_i < N){
      hidden_layer_i = hidden_layer_i+1
      model %>%
        layer_dropout(rate = 0.4) %>%
        layer_dense(units = hidden_layers[hidden_layer_i], activation = activation)
    }
    
    # Output layer
    model %>%
      layer_dense(units = units_output, activation = 'softmax')
    
    # Compile the model
    model %>% compile(
      optimizer = 'adam', 
      loss = 'mean_squared_error',
      metrics = c('accuracy')
    )

    # Fit the model
    model %>% fit(x_train, y_train, epochs = epochs)
    
    # Evaluate accuracy
    score <- model %>% evaluate(x_test, y_test)
    cat('Test loss:', score$loss, "\n")
    cat('Test accuracy:', score$acc, "\n")
  }
}
