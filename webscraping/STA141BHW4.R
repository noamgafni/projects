library(httr)
library(jsonlite)
library(RCurl)
library(tidyverse)
library(RSelenium)
library(XML)
library(stopwords)
library(ggplot2)
library(dplyr)

#TASK 1

#found the URL from the json file containing all of the searched jobs
url <- "https://jobs-in-data.com/msearch/indexes/job/search"
headers <- add_headers(
  `Content-Type` = "application/json",
  `User-Agent` = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:138.0) Gecko/20100101 Firefox/138.0",
  `Authorization` = "Bearer f8eea7876c3f3ba38b2980bd52aa864fa0eba1e9365bd9f67c8b11f35e9b1366"
)

jobscrape <- function(query) {
  body <- list(
    q = query,
    #make sure we get the most recent searches
    sort = list("date:desc"),
    limit = 1000,
    offset = 0
  )
  
  json_body <- toJSON(body, auto_unbox = TRUE)
  
  response <- POST(url, headers, body = json_body, encode = "raw")
  txt <- content(response, as = "text", encoding = "UTF-8")
  llist <- fromJSON(txt)
  #found hits from exploring the llist nested list
  hits = llist$hits
  hits$salary_avg <- as.numeric(hits$salary_avg)
  hits$job = query
  return(hits)
}

###SELEIUM HTML EXTRACTION
#pass 1
extract_html = function(url_list) {
  dr = remoteDriver$new()
  dr$open()
  
  # Wait a second or two for the browser to dynamically build the page's contents
  p = dr$getPageSource()
  #doc = htmlParse(p[[1]])
  html = lapply(url_list,
                function(u) {
                  dr$navigate(u)
                  # Wait a second or two for the browser to dynamically build the page's contents
                  #Waiting 10 seconds in order to let page fully load (took a lot of time...)
                  Sys.sleep(10)
                  p = dr$getPageSource()[[1]]
                })
}


#pass 2
#chat gpt helped me figure out how to bypass "accept cookies prompt"
extract_html2 <- function(url_list) {
  dr <- remoteDriver$new()
  dr$open()
  
  html <- lapply(url_list, function(u) {
    dr$navigate(u)
    Sys.sleep(4)  # Let the page load
    
    # I looked through some websites that have cookie prompts and foun the most common phrases to accept
    cookie_xpaths <- c(
      "//button[contains(translate(., 'ACEPT', 'acept'), 'accept')]",
      "//button[contains(translate(., 'OK', 'ok'), 'ok')]",
      "//button[contains(translate(., 'AGREE', 'agree'), 'agree')]",
      "//div[contains(@class, 'cookie') or contains(@id, 'cookie')]//button",
      "//button[contains(text(), 'Got it')]",
      "//button[contains(text(), 'Accept all')]",
      "//a[contains(text(), 'Accept')]"
    )
    
    for (xpath in cookie_xpaths) {
      try({
        el <- dr$findElement("xpath", xpath)
        el$clickElement()
        Sys.sleep(2)
        break  #break if a button works
      }, silent = TRUE)
    }
    
    Sys.sleep(4)  # Wait for the rest of the content to load
    p <- dr$getPageSource()[[1]]
    return(p)
  })
}


###TASK 2 PART 1
extract_section_titles <- function(html_list) {
  doc <- htmlParse(html_list, asText = TRUE)
  nodes <- getNodeSet(doc, "//h1|//h2|//h3|//h4|//h5|//h6|//b|//strong")
  titles <- sapply(nodes, xmlValue)
  #make the titles more general so that we can combine similar titles later
  titles <- trimws(titles)
  titles <- tolower(titles)
  #removing all punctuation from the words
  titles <- gsub("[[:punct:]]", "", titles) 
  titles <- titles[nchar(titles) > 2 & nchar(titles) < 75]
  return(titles)
}


#these are the keywords that I found by randomly clicking through the job listings and looking at the headers
keywords <- c("responsibilities", "qualifications", "about us", "education", "skills", "experience", "descrption")

combine_similar_titles <- function(titles, keywords) {
  #\\b will be able to find key words in their different forms and collapse them into one category 
  pattern <- paste0("\\b(", paste(keywords, collapse = "|"), ")\\b")
  cleaned_titles <- unlist(lapply(titles, function(title) {
    match <- regmatches(title, regexpr(pattern, title))
    if (length(match) > 0) {
      match
    } else {
      title
    }
  }))
  return(cleaned_titles)
}

###TASK 2 PART 2
extract_visible_text <- function(html_list) {
  doc <- htmlParse(html_list, asText = TRUE)
  removeNodes <- getNodeSet(doc, "//script | //style")
  #invisible(lapply(removeNodes, removeNodes))
  text <- xpathSApply(doc, "//body//text()", xmlValue)
  text <- paste(trimws(text), collapse = " ")
  return(text)
}

remove_stopwords <- function(text) {
  words <- tolower(unlist(strsplit(text, "\\W+")))
  words <- words[words != ""]
  words <- words[!words %in% stopwords("en")]
  #had to take out "s" since it became its own word due to contractions, as well as other 2 letter words
  words <- words[nchar(words) > 2]
  return(words)
}

#in hindsight I could have bade a function to get words_list instead of copying code between functions
get_common_words <- function(html_list) {
  texts <- unlist(lapply(html_list, function(html) {
    sections <- extract_section_titles(html)
    unlist(sections)
  }))
  words_list <- lapply(texts, remove_stopwords)
  all_words <- unlist(words_list)
  head(sort(table(all_words), decreasing = TRUE), 50)
}

get_common_phrases <- function(html_list, exclude_words = c("job", "apply", "posting", "storage")) {
  texts <- unlist(lapply(html_list, function(html) {
    sections <- extract_section_titles(html)
    unlist(sections)
  }))
  
  words_list <- lapply(texts, remove_stopwords)
  
  #makes sure that the length of the phrase is not longer than n, and then extracts the phrases
  get_ngrams <- function(words, n) {
    if (length(words) < n) return(character(0))
    sapply(seq_len(length(words) - n + 1), function(i)
      paste(words[i:(i+n-1)], collapse = " "))
  }
  
  phrases <- unlist(lapply(words_list, function(words)
    #look for phrases between 2 and 5 words long
    unlist(lapply(2:5, function(n) get_ngrams(words, n)))))
  
  phrase_tbl <- sort(table(phrases), decreasing = TRUE)
  
  # remove excluded words after getting the initial phrases
  keep <- !sapply(names(phrase_tbl), function(p) {
    any(sapply(exclude_words, function(w) grepl(paste0("\\b", w, "\\b"), p)))
  })
  
  head(phrase_tbl[keep], 25)
}

