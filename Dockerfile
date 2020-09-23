##########################################
# Installing Debian packages
##########################################

# Install R version 3.6.1
FROM rocker/r-ver:4.0.2

# dos2unix fix for windows
RUN apt-get update && apt-get install -y dos2unix

# Install shiny server dependencies
RUN apt-get update && apt-get install -y \
    sudo \
    gdebi-core \
    pandoc \
    pandoc-citeproc \
    libcurl4-gnutls-dev \
    libcairo2-dev \
    libxt-dev \
    xtail \
    wget \
    tesseract-ocr

# Download and install shiny server
RUN wget --no-verbose https://download3.rstudio.org/ubuntu-14.04/x86_64/VERSION -O "version.txt" && \
    VERSION=$(cat version.txt)  && \
    wget --no-verbose "https://download3.rstudio.org/ubuntu-14.04/x86_64/shiny-server-$VERSION-amd64.deb" -O ss-latest.deb && \
    gdebi -n ss-latest.deb && \
    rm -f version.txt ss-latest.deb && \
    . /etc/environment && \
    R -e "install.packages(c('shiny', 'rmarkdown'), repos='https://cran.ma.imperial.ac.uk/')" && \
    cp -R /usr/local/lib/R/site-library/shiny/examples/* /srv/shiny-server/ && \
    chown shiny:shiny /var/lib/shiny-server

# Install Python 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev

##########################################
# Installing R and Python dependencies
##########################################

# R
COPY /app/requirement.R /tmp/Requirements.R
RUN Rscript /tmp/Requirements.R

# Python
COPY python/requirement.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN python3 -m nltk.downloader stopwords

##########################################
# Deploying R and Python Code
##########################################

# Copy code
COPY /app /srv/shiny-server/
COPY /python /srv/python/
COPY /data /srv/data/

# Copy further configuration files into the Docker image and run
COPY config.yaml /srv/config.yaml
COPY shiny-server.conf  /etc/shiny-server/shiny-server.conf
COPY shiny-server.sh /usr/bin/shiny-server.sh

RUN mkdir srv/outputs
RUN mkdir srv/outputs/models
RUN mkdir srv/outputs/features
RUN chmod -R 777 srv/*

# Make the ShinyApp available at port 3838
EXPOSE 3838

# dos2unix fix for Windows
RUN dos2unix /usr/bin/shiny-server.sh && apt-get --purge remove -y dos2unix && rm -rf /var/lib/apt/lists/*

# CMD python3

ENTRYPOINT ["/usr/bin/shiny-server.sh"]