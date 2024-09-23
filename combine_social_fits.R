combine_social_fits <- function(root, result_dir){
  rooms = c('Like', 'Dislike')  
  for(room in rooms){
    all.files <- list.files(glue("{root}"), pattern=glue(".*{room}.*.csv"), full.names = T)
    name <- paste("df", sep='.', room)
    assign(name, all.files %>% 
             pblapply(., FUN = read.csv) %>% 
             do.call(rbind, .) %>%
             mutate(room_type = room) %>%
             relocate(room_type, .after = session))
  } 
  df <- rbind(df.Like, df.Dislike)
  write.csv(df, glue("{result_dir}/fits_{Sys.Date()}.csv"), row.names=F)
}