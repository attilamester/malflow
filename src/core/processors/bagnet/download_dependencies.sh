log() {
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] [Download deps.] $1"
}

download_file() {
    if [ $# -ne 2 ]; then
        log "Error: Please provide exactly two parameters for the download_file function"
        exit 1
    fi
    log "Downloading to [$1] from [$2]"
    res=`wget --quiet --no-verbose --show-progress -O $1 $2`
    if [ $? -eq 0 ]; then
        log "Downloaded successfully."
    else
        log "Error: Download failed."
        exit 1
    fi
}

if [ $# -ne 1 ]; then
    log "Usage: $0 <path-to-main.py>"
    exit 1
fi

download_file $1 https://raw.githubusercontent.com/attilamester/pytorch-examples/feature/83/imagenet/main.py?v=227a4533
